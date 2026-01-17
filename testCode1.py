import cv2
import numpy as np
from ultralytics import YOLO
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.models.video as video_models
from collections import deque
import time

# Ask user for vertical line X position BEFORE starting anything
print("="*50)
print("SMART TRAFFIC MANAGEMENT SYSTEM")
print("="*50)
print("This program will draw a vertical line at your specified X coordinate")
print("and count vehicles that cross this line.")
print()

while True:
    try:
        vertical_line_x = int(input("Enter the X coordinate for the vertical line (e.g., 300): "))
        print(f"\nVertical line will be drawn at X = {vertical_line_x}")
        break
    except ValueError:
        print("Please enter a valid number!")

print("\nStarting camera... Press 'q' to quit")
print("-" * 50)

# Initialize YOLO model
model = YOLO("yolov8n.pt")

vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck
ROI_TOP_LEFT = (50, 50)
ROI_BOTTOM_RIGHT = (600, 600)

# Start video capture AFTER getting user input
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera stream")
    input("Press Enter to exit...")
    exit()

# Accident detection integration
class TrafficAccidentDetector:
    def __init__(self):
        # Initializing 'self.frame_buffer' here is crucial for the class to work.
        self.model = self.load_pretrained_model()
        self.frame_buffer = deque(maxlen=32)  # Store 32 frames (~1-2s at 30 FPS)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            # Normalization is often done with mean/std of the dataset the model was trained on
            transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225]),
        ])
        self.fps = 30  # Assume 30 FPS; adjust based on cap.get(cv2.CAP_PROP_FPS)
    
    def load_pretrained_model(self):
        # Load SlowFast model via TorchHub from pytorchvideo repository
        try:
            m = torch.hub.load("facebookresearch/pytorchvideo", "slowfast_r50", pretrained=True)
            num_classes = 5
            # Adjust the final layer for 5 custom classes
            m.blocks[-1].proj = nn.Linear(m.blocks[-1].proj.in_features, num_classes)
        except:
            # Fallback if torch.hub.load fails or is not available
            m = video_models.slowfast_r50(weights=video_models.SlowFast_R50_Weights.DEFAULT)
            num_classes = 5  # normal_traffic, accident, near_miss, emergency_brake, wrong_direction
            m.fc = nn.Linear(m.fc.in_features, num_classes)
        m.eval()
        return m
    
    def prepare_clip(self):
        # Convert list of CHW tensors to TCHW, apply transforms
        # Note: self.transform is designed for a batch of TCHW, so torch.stack must be TCHW
        # Assuming the transform can handle a TCHW tensor (it expects 3D or 4D input)
        clip = torch.stack(list(self.frame_buffer))  # TCHW
        # The provided transform will likely require adjustment or batching if it expects C*H*W*
        # Simplification: Apply transform *per frame* before stacking, or adjust transform
        # For simplicity and to match common video processing:
        # We will assume a simple resize/normalize on the stacked TCHW clip for now, 
        # which may need adjustment based on specific model requirements.
        
        # Prepare for normalization and model input (SlowFast requires special input format, but 
        # for this general structure, we stack and transform)
        clip_list = [self.transform(frame_tensor) for frame_tensor in list(self.frame_buffer)]
        clip = torch.stack(clip_list) # TCHW
        return clip.unsqueeze(0)  # Add Batch dim: BTCHW -> B*T*C*H*W
    
    def process_opencv_input(self, frame, vehicle_data):
        # Convert BGR OpenCV frame to CHW [0,1] float Tensor
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0  # CHW [0,1]
        
        # Append to the detector's internal buffer
        self.frame_buffer.append(frame_tensor)
        
        accident_status = None
        # Only run inference when the buffer is full (e.g., 32 frames)
        if len(self.frame_buffer) == 32:
            clip = self.prepare_clip()
            with torch.no_grad():
                # Note: SlowFast models typically require two clips (slow/fast), 
                # but for general integration, we use a single clip input format.
                prediction = self.model(clip) 
            # Analyze prediction and vehicle data
            accident_status = self.analyze_prediction(prediction, vehicle_data)
            
            # Optional: Clear buffer after inference to start collecting the next clip, 
            # or keep it as a sliding window (not clearing is a sliding window)
            # self.frame_buffer.clear() 
            
        return accident_status
    
    def analyze_prediction(self, model_output, vehicle_data):
        ai_confidence = torch.softmax(model_output, dim=1)
        # Assuming index 1 is 'accident' and 2 is 'near_miss'
        
        # Vehicle-based heuristics from YOLO tracking
        collision_detected = vehicle_data.get('collision', False)
        sudden_speed_change = vehicle_data.get('speed_change', 0) > 20
        
        if ai_confidence[0][1] > 0.7 and (collision_detected or sudden_speed_change):
            return "ACCIDENT_DETECTED"
        elif ai_confidence[0][2] > 0.6:
            return "NEAR_MISS"
        else:
            # Check if AI model suggests normal traffic with high confidence
            if ai_confidence[0][0] > 0.8: # Assuming index 0 is 'normal_traffic'
                 return "NORMAL"
            return None # No strong signal yet

# Initialize accident detector
accident_detector = TrafficAccidentDetector()

# Removed: frame_buffer = deque(maxlen=90) # This buffer is redundant
last_positions = {}  # Track vehicle positions for speed/collision

def point_in_roi(x, y, top_left, bottom_right):
    return top_left[0] <= x <= bottom_right[0] and top_left[1] <= y <= bottom_right[1]

# Removed: extract_clip function

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to read from camera stream")
        break

    results = model(frame, stream=True)
    
    vehicle_count = 0
    vehicles_crossing_line = 0
    lanes = {0: [], 1: [], 2: []}
    
    # Check if frame has valid dimensions before calculating lane_width
    if frame.shape[1] > ROI_TOP_LEFT[0]:
        lane_width = (ROI_BOTTOM_RIGHT[0] - ROI_TOP_LEFT[0]) // 3
    else:
        lane_width = 100 # Default if frame is too small

    cv2.rectangle(frame, ROI_TOP_LEFT, ROI_BOTTOM_RIGHT, (255, 255, 0), 2)

    # Vehicle tracking for collision/speed
    current_positions = {}
    vehicle_data = {'collision': False, 'speed_change': 0}
    
    for r in results:
        if hasattr(r, 'boxes') and r.boxes is not None:
            for box in r.boxes:
                # Safely extract box attributes
                cls = int(box.cls[0]) if hasattr(box, 'cls') and box.cls is not None and len(box.cls) > 0 else None
                conf = float(box.conf[0]) if hasattr(box, 'conf') and box.conf is not None and len(box.conf) > 0 else 0
                coords = box.xyxy[0] if hasattr(box, 'xyxy') and box.xyxy is not None and len(box.xyxy) > 0 else None
                box_id = int(box.id[0]) if hasattr(box, 'id') and box.id is not None and len(box.id) > 0 else None

                if cls in vehicle_classes and coords is not None and conf > 0.25:
                    x1, y1, x2, y2 = map(int, coords)
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2

                    if point_in_roi(cx, cy, ROI_TOP_LEFT, ROI_BOTTOM_RIGHT):
                        vehicle_count += 1
                        lane_index = (cx - ROI_TOP_LEFT[0]) // lane_width
                        lane_index = max(0, min(2, lane_index))
                        lanes[lane_index].append((cx, cy))

                        # Track position for speed/collision
                        if box_id is not None:
                            current_positions[box_id] = (cx, cy)
                        
                        if x1 <= vertical_line_x <= x2:
                            vehicles_crossing_line += 1
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                        else:
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                        label = model.names[cls] if hasattr(model, 'names') and cls is not None and cls < len(model.names) else str(cls)
                        cv2.putText(frame, label, (x1, y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Calculate vehicle_data for accident detection
    for box_id, (cx, cy) in current_positions.items():
        if box_id in last_positions:
            last_cx, last_cy = last_positions[box_id]
            distance = ((cx - last_cx) ** 2 + (cy - last_cy) ** 2) ** 0.5
            # Speed is distance * fps. The scale is in pixels, not real-world units.
            speed = distance * accident_detector.fps 
            vehicle_data['speed_change'] = max(vehicle_data['speed_change'], speed)
            
            # Check for collision (e.g., vehicles too close)
            for other_id, (ocx, ocy) in current_positions.items():
                if box_id != other_id:
                    # Very simple collision check based on center distance
                    if ((cx - ocx) ** 2 + (cy - ocy) ** 2) ** 0.5 < 50:  # 50px threshold
                        vehicle_data['collision'] = True
                        break # Collision detected for this vehicle
    last_positions = current_positions

    # --- ACCIDENT DETECTION CALL ---
    # This is called on every frame to let the detector manage its internal buffer.
    # It will only run the heavy model inference when its buffer is full (every 32 frames).
    accident_status = accident_detector.process_opencv_input(frame, vehicle_data)

    # Display accident status
    if accident_status:
        color = (0, 0, 255) if "ACCIDENT" in accident_status else (0, 255, 255) if "MISS" in accident_status else (255, 0, 0)
        cv2.putText(frame, f"Status: {accident_status}", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

    queue_length = vehicles_crossing_line
    
    # Ensure vertical_line_x is within frame bounds for drawing
    if 0 <= vertical_line_x < frame.shape[1]:
        cv2.line(frame, (vertical_line_x, 0), (vertical_line_x, frame.shape[0]), (255, 0, 0), 4)
        cv2.putText(frame, f"X = {vertical_line_x}", (vertical_line_x + 10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    for i, (lane_id, vehicles) in enumerate(lanes.items()):
        lane_count = len(vehicles)
        if lane_count > 0:
            cv2.putText(frame, f"Lane {lane_id + 1}: {lane_count}", (20, 160 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.putText(frame, f"Total Vehicles: {vehicle_count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
    cv2.putText(frame, f"Queue Length: {queue_length}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    cv2.imshow("Traffic Management - Vertical Line Detection", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("\nExiting program...")
        break

cap.release()
cv2.destroyAllWindows()
print("Program closed.")       
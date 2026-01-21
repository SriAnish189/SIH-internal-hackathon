import cv2
import torch
from ultralytics import YOLO
from collections import deque
import torchvision.transforms as T
from torchvision.models.video import r3d_18, R3D_18_Weights
import torch.nn as nn
import time
import math
import os

# ------------------- USER PARAMETERS -------------------
YOLO_WEIGHTS_PATH = "yolov8n.pt"  # Path to YOLOv8 weights
CAMERA_INDEX = 0
VERTICAL_LINE_X = int(input("Enter vertical line X coordinate (e.g., 300): "))
ROI_TOP_LEFT = (50, 50)
ROI_BOTTOM_RIGHT = (600, 600)
VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck
FRAME_BUFFER_LEN = 150           # enough frames to save a 5s clip
SPEED_THRESHOLD = 20
COLLISION_THRESHOLD = 50
ACCIDENT_CLIP_DURATION = 90      # save 90 frames (~3s) after detection
OUTPUT_FOLDER = "accident_clips"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

print("\nStarting camera... Press 'q' to quit")

# ------------------- YOLO MODEL -----------------------
model = YOLO(YOLO_WEIGHTS_PATH)

# ---------------- ACCIDENT DETECTOR -------------------
class AccidentDetector:
    def __init__(self):
        self.model = self.load_model()
        self.frame_buffer = deque(maxlen=32)  # store 32 frames for AI clip
        self.transform = T.Compose([
            T.ConvertImageDtype(torch.float32),
            T.Resize((112, 112)),
            T.Normalize(mean=[0.43216, 0.394666, 0.37645],
                        std=[0.22803, 0.22145, 0.216989])
        ])
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()

    def load_model(self):
        m = r3d_18(weights=R3D_18_Weights.DEFAULT)
        m.fc = nn.Linear(m.fc.in_features, 2)  # accident / no accident
        return m

    def process_frame(self, frame):
    # Convert frame (H,W,C) -> (C,H,W)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1) / 255.0
        self.frame_buffer.append(frame_tensor)

    # Run AI only when we have 32 frames
        if len(self.frame_buffer) == 32:
        # Stack frames: [T, C, H, W]
            clip = torch.stack(list(self.frame_buffer))

        # Apply transform frame-wise
            transformed_frames = []
            for i in range(clip.shape[0]):  # for each frame in clip
                img = clip[i]  # shape: [3, H, W]
                img = self.transform(img)  # Apply resize + normalize
                transformed_frames.append(img)
        
        # Stack back: [T, C, H, W]
            clip = torch.stack(transformed_frames)

        # Permute to [C, T, H, W]
            clip = clip.permute(1, 0, 2, 3)

        # Add batch dimension: [1, C, T, H, W]
            clip = clip.unsqueeze(0).to(self.device)

            with torch.no_grad():
                pred = self.model(clip)
                conf = torch.softmax(pred, dim=1)
                if conf[0][1] > 0.7:  # class 1 = accident
                    return "ACCIDENT_DETECTED"
        return None


accident_detector = AccidentDetector()

# ------------------- VIDEO CAPTURE --------------------
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print("Error: Cannot open camera")
    exit()

prev_positions = {}
vehicle_id_counter = 0
frame_buffer = deque(maxlen=FRAME_BUFFER_LEN)  # For saving accident clips
accident_flag = False  # prevents multiple saves per event

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    vehicle_count = 0
    vehicles_crossing_line = 0
    lanes = {0: [], 1: [], 2: []}
    lane_width = (ROI_BOTTOM_RIGHT[0] - ROI_TOP_LEFT[0]) // 3
    cv2.rectangle(frame, ROI_TOP_LEFT, ROI_BOTTOM_RIGHT, (255, 255, 0), 2)

    current_positions = {}
    collisions_detected = False
    max_speed_change = 0

    # ------------------- YOLO DETECTION -------------------
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            if cls not in VEHICLE_CLASSES:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            if ROI_TOP_LEFT[0] <= cx <= ROI_BOTTOM_RIGHT[0] and ROI_TOP_LEFT[1] <= cy <= ROI_BOTTOM_RIGHT[1]:
                vehicle_count += 1
                lane_idx = min(2, max(0, (cx - ROI_TOP_LEFT[0]) // lane_width))
                lanes[lane_idx].append((cx, cy))

                # Simple ID assignment
                assigned_id = None
                for vid, (px, py) in prev_positions.items():
                    if math.hypot(cx - px, cy - py) < 50:
                        assigned_id = vid
                        break
                if assigned_id is None:
                    vehicle_id_counter += 1
                    assigned_id = vehicle_id_counter
                current_positions[assigned_id] = (cx, cy)

                # Speed calculation
                if assigned_id in prev_positions:
                    px, py = prev_positions[assigned_id]
                    speed = math.hypot(cx - px, cy - py)
                    max_speed_change = max(max_speed_change, speed)

                # Collision detection
                for other_id, (ocx, ocy) in current_positions.items():
                    if assigned_id != other_id and math.hypot(cx - ocx, cy - ocy) < COLLISION_THRESHOLD:
                        collisions_detected = True

                color = (0, 0, 255) if x1 <= VERTICAL_LINE_X <= x2 else (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, str(cls), (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                if x1 <= VERTICAL_LINE_X <= x2:
                    vehicles_crossing_line += 1

    prev_positions = current_positions

    # ---------------- ACCIDENT DETECTION ----------------
    accident_status = accident_detector.process_frame(frame)
    if collisions_detected or max_speed_change > SPEED_THRESHOLD:
        accident_status = "ACCIDENT_DETECTED"

    # ---------------- FRAME BUFFER FOR CLIP ----------------
    frame_buffer.append(frame.copy())

    if accident_status == "ACCIDENT_DETECTED" and not accident_flag:
        accident_flag = True
        timestamp = int(time.time())
        output_path = os.path.join(OUTPUT_FOLDER, f"accident_{timestamp}.mp4")
        h, w, _ = frame.shape
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h))
        for f in list(frame_buffer)[-ACCIDENT_CLIP_DURATION:]:  # last N frames
            out.write(f)
        out.release()
        print(f"[INFO] Accident clip saved: {output_path}")

    # ---------------- DISPLAY INFO ----------------
    if accident_status:
        cv2.putText(frame, f"Status: {accident_status}", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    cv2.line(frame, (VERTICAL_LINE_X, 0), (VERTICAL_LINE_X, frame.shape[0]), (255, 0, 0), 2)
    cv2.putText(frame, f"X = {VERTICAL_LINE_X}", (VERTICAL_LINE_X + 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    for i, vehicles in lanes.items():
        cv2.putText(frame, f"Lane {i+1}: {len(vehicles)}", (20, 160 + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"Total Vehicles: {vehicle_count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
    cv2.putText(frame, f"Queue Length: {vehicles_crossing_line}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    cv2.imshow("Traffic Management", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
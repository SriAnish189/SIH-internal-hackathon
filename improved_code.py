import cv2
import numpy as np
from ultralytics import YOLO
import pygame
import time
import json
from datetime import date
from collections import deque

# Constants for Pygame
WIDTH, HEIGHT = 1000, 800
ROAD_WIDTH = 120
FPS = 30
ROAD_COLOR = (50, 50, 50)  # Dark grey for roads
GRASS_COLOR = (0, 128, 0)  # Green for background
LINE_COLOR = (255, 255, 255)  # White for road lines
YELLOW_COLOR = (255, 255, 0)  # Yellow for traffic lights

# Default signal timers
MIN_GREEN_TIME = 10
MAX_GREEN_TIME = 45
DEFAULT_GREEN_TIME = 20

# Vehicle tracking for speed calculation
class VehicleTracker:
    def __init__(self):
        self.tracks = {}
        self.next_id = 0
        self.max_disappeared = 5
    
    def update(self, detections):
        if len(detections) == 0:
            # Mark all tracks as disappeared
            for track_id in list(self.tracks.keys()):
                self.tracks[track_id]['disappeared'] += 1
                if self.tracks[track_id]['disappeared'] > self.max_disappeared:
                    del self.tracks[track_id]
            return []
        
        # Simple centroid tracking
        current_centroids = []
        for detection in detections:
            x1, y1, x2, y2 = detection
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            current_centroids.append((cx, cy))
        
        # Update existing tracks or create new ones
        if len(self.tracks) == 0:
            for centroid in current_centroids:
                self.tracks[self.next_id] = {
                    'centroid': centroid,
                    'disappeared': 0,
                    'positions': deque(maxlen=10),
                    'speed': 0.0
                }
                self.tracks[self.next_id]['positions'].append(centroid)
                self.next_id += 1
        else:
            # Simple nearest neighbor assignment
            track_ids = list(self.tracks.keys())
            for i, centroid in enumerate(current_centroids):
                if i < len(track_ids):
                    track_id = track_ids[i]
                    self.tracks[track_id]['centroid'] = centroid
                    self.tracks[track_id]['positions'].append(centroid)
                    self.tracks[track_id]['disappeared'] = 0
                    
                    # Calculate speed based on position history
                    if len(self.tracks[track_id]['positions']) >= 2:
                        pos1 = self.tracks[track_id]['positions'][-2]
                        pos2 = self.tracks[track_id]['positions'][-1]
                        distance = np.sqrt((pos2[0] - pos1[0])**2 + (pos2[1] - pos1[1])**2)
                        self.tracks[track_id]['speed'] = distance * FPS / 100.0
        
        return list(self.tracks.values())

# Vehicle class for visualization
class Vehicle:
    def __init__(self, direction, pos, speed=2, vtype='car'):
        self.direction = direction  # 'NS' or 'SN'
        self.pos = list(pos)
        self.speed = speed
        self.size = (25, 40) if direction == 'NS' else (25, 40)  # Vertical orientation
        self.color = (100, 150, 255) if vtype == 'car' else (255, 100, 100) if vtype == 'bus' else (150, 150, 150)
        self.vtype = vtype

    def draw(self, screen):
        pygame.draw.rect(screen, self.color, (*self.pos, *self.size))
        # Add direction indicator
        if self.direction == 'NS':
            pygame.draw.polygon(screen, (255, 255, 255), [
                (self.pos[0] + 12, self.pos[1]),
                (self.pos[0] + 8, self.pos[1] + 10),
                (self.pos[0] + 16, self.pos[1] + 10)
            ])
        else:  # SN
            pygame.draw.polygon(screen, (255, 255, 255), [
                (self.pos[0] + 12, self.pos[1] + 40),
                (self.pos[0] + 8, self.pos[1] + 30),
                (self.pos[0] + 16, self.pos[1] + 30)
            ])

# Traffic Simulator for 2-way junction
class TwoWayTrafficSimulator:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Smart 2-Way Traffic Junction - N-S vs S-N")
        self.clock = pygame.time.Clock()
        
        # Traffic data
        self.ns_vehicles = []  # North to South vehicles
        self.sn_vehicles = []  # South to North vehicles
        
        # Signal states: True = Green, False = Red
        self.ns_green = True   # N-S direction signal
        self.sn_green = False  # S-N direction signal
        
        # Traffic metrics
        self.ns_queue_length = 0
        self.sn_queue_length = 0
        self.ns_avg_speed = 1.0
        self.sn_avg_speed = 1.0
        
        # Total vehicle counts (for visualization)
        self.ns_vehicle_count = 0
        self.sn_vehicle_count = 0
        
        # Signal timing
        self.signal_timer = time.time()
        self.current_green_time = DEFAULT_GREEN_TIME
        self.running = True
        
        # Fonts
        self.font = pygame.font.Font(None, 28)
        self.large_font = pygame.font.Font(None, 40)
        
        print("2-Way Traffic Junction Simulator Initialized")

    def update_traffic_data(self, ns_data, sn_data):
        """Update traffic data from both cameras"""
        self.ns_queue_length = ns_data['queue_length']
        self.ns_avg_speed = max(0.1, ns_data['avg_speed'])
        self.ns_vehicle_count = ns_data['vehicle_count']
        
        self.sn_queue_length = sn_data['queue_length']
        self.sn_avg_speed = max(0.1, sn_data['avg_speed'])
        self.sn_vehicle_count = sn_data['vehicle_count']
        
        # Update vehicle visualization using total vehicle counts
        self.update_vehicles()
        
        print(f"Traffic Update - N-S Queue={self.ns_queue_length}, N-S Vehicles={self.ns_vehicle_count}, Speed={self.ns_avg_speed:.1f} | "
              f"S-N Queue={self.sn_queue_length}, S-N Vehicles={self.sn_vehicle_count}, Speed={self.sn_avg_speed:.1f}")

    def update_vehicles(self):
        """Update vehicle positions for visualization using total vehicle counts"""
        # Clear old vehicles
        self.ns_vehicles = []
        self.sn_vehicles = []
        
        # Add N-S vehicles (based on total vehicles count)
        for i in range(min(8, self.ns_vehicle_count)):
            pos = [WIDTH//2 - 60, 100 + i * 50]
            self.ns_vehicles.append(Vehicle('NS', pos))
        
        # Add S-N vehicles (based on total vehicles count) 
        for i in range(min(8, self.sn_vehicle_count)):
            pos = [WIDTH//2 + 40, HEIGHT - 100 - i * 50]
            self.sn_vehicles.append(Vehicle('SN', pos))

    def calculate_optimal_timing(self):
        """AI-based adaptive signal timing"""
        # Calculate traffic demand (queue length / speed ratio)
        ns_demand = self.ns_queue_length / max(0.1, self.ns_avg_speed)
        sn_demand = self.sn_queue_length / max(0.1, self.sn_avg_speed)
        
        total_demand = ns_demand + sn_demand
        
        if total_demand > 0:
            # Calculate proportional green times
            ns_ratio = ns_demand / total_demand
            sn_ratio = sn_demand / total_demand
            
            # Determine next green time based on higher demand
            if self.ns_green:  # Currently N-S is green
                # Next will be S-N, calculate S-N green time
                next_green_time = MIN_GREEN_TIME + (sn_ratio * (MAX_GREEN_TIME - MIN_GREEN_TIME))
            else:  # Currently S-N is green
                # Next will be N-S, calculate N-S green time
                next_green_time = MIN_GREEN_TIME + (ns_ratio * (MAX_GREEN_TIME - MIN_GREEN_TIME))
            
            return max(MIN_GREEN_TIME, min(MAX_GREEN_TIME, next_green_time))
        
        return DEFAULT_GREEN_TIME

    def update_signals(self):
        """Update traffic signal timing"""
        current_time = time.time()
        elapsed = current_time - self.signal_timer
        
        # Check if it's time to switch signals
        if elapsed >= self.current_green_time:
            # Switch signals
            self.ns_green = not self.ns_green
            self.sn_green = not self.sn_green
            
            # Calculate next green time based on traffic conditions
            self.current_green_time = self.calculate_optimal_timing()
            
            # Reset timer
            self.signal_timer = current_time
            
            # Log signal change
            current_direction = "N-S" if self.ns_green else "S-N"
            print(f"Signal Changed: {current_direction} GREEN for {self.current_green_time:.1f}s")

    def draw_road_infrastructure(self):
        """Draw the 2-way road infrastructure"""
        # Fill background
        self.screen.fill(GRASS_COLOR)
        
        # Draw main vertical road
        road_x = WIDTH//2 - ROAD_WIDTH//2
        pygame.draw.rect(self.screen, ROAD_COLOR, (road_x, 0, ROAD_WIDTH, HEIGHT))
        
        # Draw center divider line
        divider_x = WIDTH//2
        for y in range(0, HEIGHT, 40):
            pygame.draw.rect(self.screen, YELLOW_COLOR, (divider_x - 2, y, 4, 20))
        
        # Draw road edges
        pygame.draw.line(self.screen, LINE_COLOR, (road_x, 0), (road_x, HEIGHT), 3)
        pygame.draw.line(self.screen, LINE_COLOR, (road_x + ROAD_WIDTH, 0), (road_x + ROAD_WIDTH, HEIGHT), 3)
        
        # Draw junction area (middle section)
        junction_y = HEIGHT//2 - 80
        junction_height = 160
        pygame.draw.rect(self.screen, ROAD_COLOR, (road_x, junction_y, ROAD_WIDTH, junction_height))
        
        # Draw U-turn arrows in junction
        arrow_color = (255, 255, 150)
        # N-S U-turn arrow
        pygame.draw.arc(self.screen, arrow_color, (WIDTH//2 - 40, HEIGHT//2 - 20, 40, 40), 0, 3.14, 3)
        # S-N U-turn arrow  
        pygame.draw.arc(self.screen, arrow_color, (WIDTH//2, HEIGHT//2 - 20, 40, 40), 3.14, 6.28, 3)

    def draw_traffic_lights(self):
        """Draw traffic lights for both directions"""
        # N-S Traffic Light (Left side)
        ns_light_x = WIDTH//2 - 80
        ns_light_y = HEIGHT//2 - 60
        
        # Light box
        pygame.draw.rect(self.screen, (0, 0, 0), (ns_light_x - 15, ns_light_y - 40, 30, 80))
        
        # Red light
        red_color = (255, 0, 0) if not self.ns_green else (100, 0, 0)
        pygame.draw.circle(self.screen, red_color, (ns_light_x, ns_light_y - 20), 10)
        
        # Green light
        green_color = (0, 255, 0) if self.ns_green else (0, 100, 0)
        pygame.draw.circle(self.screen, green_color, (ns_light_x, ns_light_y + 20), 10)
        
        # Label
        label = self.font.render("N→S", True, (255, 255, 255))
        self.screen.blit(label, (ns_light_x - 20, ns_light_y + 50))
        
        # S-N Traffic Light (Right side)
        sn_light_x = WIDTH//2 + 80
        sn_light_y = HEIGHT//2 - 60
        
        # Light box
        pygame.draw.rect(self.screen, (0, 0, 0), (sn_light_x - 15, sn_light_y - 40, 30, 80))
        
        # Red light
        red_color = (255, 0, 0) if not self.sn_green else (100, 0, 0)
        pygame.draw.circle(self.screen, red_color, (sn_light_x, sn_light_y - 20), 10)
        
        # Green light
        green_color = (0, 255, 0) if self.sn_green else (0, 100, 0)
        pygame.draw.circle(self.screen, green_color, (sn_light_x, sn_light_y + 20), 10)
        
        # Label
        label = self.font.render("S→N", True, (255, 255, 255))
        self.screen.blit(label, (sn_light_x - 20, sn_light_y + 50))

    def draw_info_panel(self):
        """Draw information panel"""
        # Background for info panel
        panel_width = 350
        panel_height = 250
        info_rect = pygame.Rect(WIDTH - panel_width - 10, 10, panel_width, panel_height)
        pygame.draw.rect(self.screen, (0, 0, 0, 180), info_rect)
        pygame.draw.rect(self.screen, (255, 255, 255), info_rect, 2)
        
        # Title
        title = self.large_font.render("Traffic Control Status", True, (255, 255, 255))
        self.screen.blit(title, (WIDTH - panel_width, 25))
        
        # Current signal status
        current_signal = "N→S GREEN" if self.ns_green else "S→N GREEN"
        signal_color = (0, 255, 0) if self.ns_green else (255, 100, 0)
        signal_text = self.font.render(f"Current: {current_signal}", True, signal_color)
        self.screen.blit(signal_text, (WIDTH - panel_width, 70))
        
        # Time remaining
        elapsed = time.time() - self.signal_timer
        remaining = max(0, self.current_green_time - elapsed)
        time_text = self.font.render(f"Time Remaining: {remaining:.1f}s", True, (255, 255, 255))
        self.screen.blit(time_text, (WIDTH - panel_width, 100))
        
        # Traffic data (using queue length still for signal logic)
        ns_text = f"N→S: Queue={self.ns_queue_length}, Vehicles={self.ns_vehicle_count}, Speed={self.ns_avg_speed:.1f}"
        sn_text = f"S→N: Queue={self.sn_queue_length}, Vehicles={self.sn_vehicle_count}, Speed={self.sn_avg_speed:.1f}"
        
        ns_color = (0, 255, 0) if self.ns_green else (255, 255, 255)
        sn_color = (0, 255, 0) if self.sn_green else (255, 255, 255)
        
        ns_surface = self.font.render(ns_text, True, ns_color)
        sn_surface = self.font.render(sn_text, True, sn_color)
        
        self.screen.blit(ns_surface, (WIDTH - panel_width, 140))
        self.screen.blit(sn_surface, (WIDTH - panel_width, 170))
        
        # Traffic demand analysis
        ns_demand = self.ns_queue_length / max(0.1, self.ns_avg_speed)
        sn_demand = self.sn_queue_length / max(0.1, self.sn_avg_speed)
        
        demand_text = f"Demand - N→S: {ns_demand:.1f}, S→N: {sn_demand:.1f}"
        demand_surface = self.font.render(demand_text, True, (255, 255, 150))
        self.screen.blit(demand_surface, (WIDTH - panel_width, 200))

    def run_one_step(self):
        """Run one simulation step"""
        dt = self.clock.tick(FPS)
        
        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return False
        
        # Update signal timing
        self.update_signals()
        
        # Draw everything
        self.draw_road_infrastructure()
        
        # Draw vehicles using total vehicle counts
        for vehicle in self.ns_vehicles:
            vehicle.draw(self.screen)
        for vehicle in self.sn_vehicles:
            vehicle.draw(self.screen)
        
        # Draw traffic lights
        self.draw_traffic_lights()
        
        # Draw info panel
        self.draw_info_panel()
        
        pygame.display.flip()
        return self.running

    def close(self):
        """Clean up pygame"""
        pygame.quit()

# Camera processor for single direction
class DirectionalCameraProcessor:
    def __init__(self, camera_id, direction_name):
        self.camera_id = camera_id
        self.direction_name = direction_name  # "NS" or "SN"
        self.model = YOLO("yolov8n.pt")
        self.vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck
        self.tracker = VehicleTracker()
        self.frame = None
        self.last_data = {
            'queue_length': 0,
            'avg_speed': 1.0,
            'vehicle_count': 0
        }
    
    def process_frame(self, frame):
        """Process frame and extract traffic data for one direction"""
        if frame is None:
            return self.last_data
        
        self.frame = frame
        height, width = frame.shape[:2]
        
        # Run YOLO detection
        results = self.model(frame, stream=True, verbose=False)
        
        detections = []
        for r in results:
            if hasattr(r, 'boxes') and r.boxes is not None:
                for box in r.boxes:
                    cls = int(box.cls[0]) if hasattr(box, 'cls') else None
                    conf = float(box.conf[0]) if hasattr(box, 'conf') else 0
                    coords = box.xyxy[0] if hasattr(box, 'xyxy') else None
                    
                    if cls in self.vehicle_classes and coords is not None and conf > 0.3:
                        x1, y1, x2, y2 = map(int, coords)
                        detections.append((x1, y1, x2, y2))
                        
                        # Draw detection for debugging
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f'Vehicle {conf:.2f}', (x1, y1-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Update tracker and get speed data
        tracked_vehicles = self.tracker.update(detections)
        
        # Calculate metrics
        queue_length = len(detections)
        speeds = [v['speed'] for v in tracked_vehicles if v['speed'] > 0]
        avg_speed = np.mean(speeds) if speeds else 1.0
        
        # Update last data
        self.last_data = {
            'queue_length': queue_length,
            'avg_speed': max(0.1, avg_speed),
            'vehicle_count': len(tracked_vehicles)
        }
        
        # Add direction info to frame
        cv2.putText(frame, f'{self.direction_name} Direction', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.putText(frame, f'Queue: {queue_length}, Speed: {avg_speed:.1f}', (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        return self.last_data

# Main Smart Traffic System
class Smart2WayTrafficSystem:
    def __init__(self):
        self.simulator = TwoWayTrafficSimulator()
        self.ns_processor = DirectionalCameraProcessor(0, "N→S")  # Camera for N-S traffic
        self.sn_processor = DirectionalCameraProcessor(1, "S→N")  # Camera for S-N traffic
        
        # Initialize video captures
        self.cap_ns = cv2.VideoCapture(0)  # Camera for N-S direction
        self.cap_sn = cv2.VideoCapture(1)  # Camera for S-N direction
        
        # Set camera properties
        for cap in [self.cap_ns, self.cap_sn]:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("Smart 2-Way Traffic System Initialized")
        print("Camera 0: Monitoring N→S traffic")
        print("Camera 1: Monitoring S→N traffic")
    
    def run(self):
        """Main execution loop"""
        print("Starting Smart 2-Way Traffic Management System...")
        print("The system will adaptively control signals based on real traffic data")
        print("Press 'q' in camera window or close pygame window to quit")
        
        while self.simulator.running:
            # Capture frames from both cameras
            ret_ns, frame_ns = self.cap_ns.read()
            ret_sn, frame_sn = self.cap_sn.read()
            
            if not ret_ns or not ret_sn:
                print("Warning: Could not read from one or both cameras")
                # Continue with dummy data for demonstration
                ns_data = {'queue_length': 2, 'avg_speed': 1.0, 'vehicle_count': 2}
                sn_data = {'queue_length': 1, 'avg_speed': 1.5, 'vehicle_count': 1}
            else:
                # Process both camera feeds
                ns_data = self.ns_processor.process_frame(frame_ns)
                sn_data = self.sn_processor.process_frame(frame_sn)
            
            # Update simulator with real traffic data
            self.simulator.update_traffic_data(ns_data, sn_data)
            
            # Run simulation step
            if not self.simulator.run_one_step():
                break
            
            # Display camera feeds if available
            if ret_ns and ret_sn:
                # Resize and combine frames for display
                display_ns = cv2.resize(frame_ns, (400, 300))
                display_sn = cv2.resize(frame_sn, (400, 300))
                
                # Combine frames horizontally
                combined_frame = np.hstack([display_ns, display_sn])
                cv2.imshow('Traffic Cameras - N→S (Left) | S→N (Right)', combined_frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup
        self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        print("Shutting down Smart Traffic System...")
        self.cap_ns.release()
        self.cap_sn.release()
        cv2.destroyAllWindows()
        self.simulator.close()
        print("System stopped successfully.")

# Main execution
if __name__ == "__main__":
    try:
        system = Smart2WayTrafficSystem()
        system.run()
    except KeyboardInterrupt:
        print("\nSystem interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cv2.destroyAllWindows()
        pygame.quit()

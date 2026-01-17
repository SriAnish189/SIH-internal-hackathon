import cv2
import numpy as np
from ultralytics import YOLO
import pygame
import time
from collections import deque
import logging

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(_name_)

# Pygame constants
WIDTH, HEIGHT = 1000, 800
ROAD_WIDTH = 120
FPS = 30
ROAD_COLOR = (50, 50, 50)
GRASS_COLOR = (0, 128, 0)
LINE_COLOR = (255, 255, 255)
YELLOW_COLOR = (255, 255, 0)

# Signal timers
MIN_GREEN_TIME = 10
MAX_GREEN_TIME = 45
DEFAULT_GREEN_TIME = 20
VEHICLE_THRESHOLD = 3  # threshold to consider lane demand significant
AMBULANCE_SWITCH_DEADLINE = 3.0  # seconds to switch green for ambulance

# Vehicle tracker
class VehicleTracker:
    def _init_(self):
        self.tracks = {}
        self.next_id = 0
        self.max_disappeared = 5

    def update(self, detections):
        """
        Simple centroid-based tracker. Returns list of track dicts.
        Each track dict: {'centroid':(x,y),'disappeared':n,'positions':deque(),'speed':float}
        """
        if len(detections) == 0:
            for track_id in list(self.tracks.keys()):
                self.tracks[track_id]['disappeared'] += 1
                if self.tracks[track_id]['disappeared'] > self.max_disappeared:
                    del self.tracks[track_id]
            return list(self.tracks.values())

        current_centroids = [((x1 + x2) // 2, (y1 + y2) // 2) for x1, y1, x2, y2 in detections]

        if not self.tracks:
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
            # naive matching: pair by index — good enough for basic demo
            track_ids = list(self.tracks.keys())
            for i, centroid in enumerate(current_centroids):
                if i < len(track_ids):
                    track_id = track_ids[i]
                    self.tracks[track_id]['centroid'] = centroid
                    self.tracks[track_id]['positions'].append(centroid)
                    self.tracks[track_id]['disappeared'] = 0
                    if len(self.tracks[track_id]['positions']) >= 2:
                        pos1, pos2 = self.tracks[track_id]['positions'][-2], self.tracks[track_id]['positions'][-1]
                        distance = np.sqrt((pos2[0] - pos1[0]) ** 2 + (pos2[1] - pos1[1]) ** 2)
                        # approximate speed (pixels per second-ish)
                        self.tracks[track_id]['speed'] = distance * FPS / 100.0
        return list(self.tracks.values())

# Vehicle visualization
class Vehicle:
    def _init_(self, direction, pos, speed=2, vtype='car'):
        self.direction = direction
        self.pos = list(pos)
        self.speed = speed
        self.size = (25, 40)
        self.color = (100, 150, 255) if vtype == 'car' else (255, 100, 100) if vtype == 'bus' else (150, 150, 150)

    def draw(self, screen):
        pygame.draw.rect(screen, self.color, (*self.pos, *self.size))
        if self.direction == 'NS':
            pygame.draw.polygon(screen, (255, 255, 255), [(self.pos[0] + 12, self.pos[1]), (self.pos[0] + 8, self.pos[1] + 10), (self.pos[0] + 16, self.pos[1] + 10)])
        else:
            pygame.draw.polygon(screen, (255, 255, 255), [(self.pos[0] + 12, self.pos[1] + 40), (self.pos[0] + 8, self.pos[1] + 30), (self.pos[0] + 16, self.pos[1] + 30)])

# 2-way traffic simulator (display + timing logic)
class TwoWayTrafficSimulator:
    def _init_(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Smart 2-Way Traffic Junction")
        self.clock = pygame.time.Clock()
        self.ns_vehicles = []
        self.sn_vehicles = []
        self.ns_green = True
        self.sn_green = False
        self.ns_queue_length = 0
        self.sn_queue_length = 0
        self.ns_avg_speed = 1.0
        self.sn_avg_speed = 1.0
        self.ns_vehicle_count = 0
        self.sn_vehicle_count = 0
        self.signal_timer = time.time()
        self.current_green_time = DEFAULT_GREEN_TIME
        self.running = True
        self.font = pygame.font.Font(None, 28)
        self.large_font = pygame.font.Font(None, 40)
        self.last_ns_vehicle_count = 0
        self.last_sn_vehicle_count = 0

        # ambulance priority state
        self.ambulance_priority = False
        self.ambulance_requested_time = None
        self.ambulance_lane = None  # 'NS' or 'SN'

        logger.info("2-Way Traffic Junction Simulator Initialized")

    def update_traffic_data(self, ns_data, sn_data):
        self.ns_queue_length = ns_data['queue_length']
        self.ns_avg_speed = max(0.1, ns_data['avg_speed'])
        self.ns_vehicle_count = ns_data['vehicle_count']
        self.sn_queue_length = sn_data['queue_length']
        self.sn_avg_speed = max(0.1, sn_data['avg_speed'])
        self.sn_vehicle_count = sn_data['vehicle_count']
        self.update_vehicles()
        logger.info(f"Traffic Update - N-S Queue={self.ns_queue_length}, Vehicles={self.ns_vehicle_count}, Speed={self.ns_avg_speed:.1f} | "
                    f"S-N Queue={self.sn_queue_length}, Vehicles={self.sn_vehicle_count}, Speed={self.sn_avg_speed:.1f}")

    def update_vehicles(self):
        # simplistic visualization: recreate vehicles based on counts (max 8 each side)
        self.ns_vehicles = []
        self.sn_vehicles = []
        for i in range(min(8, self.ns_vehicle_count)):
            pos = [WIDTH // 2 - 60, 100 + i * 50]
            self.ns_vehicles.append(Vehicle('NS', pos))
        for i in range(min(8, self.sn_vehicle_count)):
            pos = [WIDTH // 2 + 40, HEIGHT - 100 - i * 50]
            self.sn_vehicles.append(Vehicle('SN', pos))

    def calculate_optimal_timing(self):
        total_vehicles = self.ns_vehicle_count + self.sn_vehicle_count
        if total_vehicles == 0:
            return DEFAULT_GREEN_TIME
        ns_ratio = self.ns_vehicle_count / total_vehicles
        sn_ratio = self.sn_vehicle_count / total_vehicles
        if self.ns_green:
            next_green_time = MIN_GREEN_TIME + ns_ratio * (MAX_GREEN_TIME - MIN_GREEN_TIME)
        else:
            next_green_time = MIN_GREEN_TIME + sn_ratio * (MAX_GREEN_TIME - MIN_GREEN_TIME)
        return max(MIN_GREEN_TIME, min(MAX_GREEN_TIME, next_green_time))

    def request_ambulance_priority(self, lane):
        """
        External call when ambulance detected on 'NS' or 'SN' lane.
        This stores a request; switching to ambulance lane will happen
        at most within AMBULANCE_SWITCH_DEADLINE seconds.
        """
        self.ambulance_priority = True
        self.ambulance_lane = lane
        self.ambulance_requested_time = time.time()
        logger.info(f"Ambulance priority requested for {lane} at {self.ambulance_requested_time}")

    def clear_ambulance_priority(self):
        logger.info("Clearing ambulance priority")
        self.ambulance_priority = False
        self.ambulance_requested_time = None
        self.ambulance_lane = None
        # reset normal timer so dynamic processing resumes smoothly
        self.signal_timer = time.time()
        self.current_green_time = self.calculate_optimal_timing()

    def update_signals(self):
        """
        Modified logic:
        - If ambulance priority requested:
            - If requested lane is not currently green: switch to it within AMBULANCE_SWITCH_DEADLINE sec.
            - If requested lane is already green: reset signal_timer so ambulance gets full green if desired.
        - Else (normal operation):
            - If vehicle count increased above VEHICLE_THRESHOLD since last recorded, recalc current_green_time and restart timer.
            - When current green elapsed, toggle green.
            - But if lane that will become green has fewer vehicles < threshold while current lane still > threshold: keep current green (restart timer).
        """
        # Ambulance override processing
        if self.ambulance_priority and self.ambulance_lane is not None:
            now = time.time()
            # if requested lane is NS and currently SN is green → need to switch to NS
            if self.ambulance_lane == 'NS':
                target_is_ns = True
            else:
                target_is_ns = False

            # If the target is already green
            if (target_is_ns and self.ns_green) or (not target_is_ns and self.sn_green):
                # Keep green for the ambulance - refresh timer so it doesn't expire during ambulance crossing
                self.signal_timer = now
                # optionally extend green time to a short window; we'll keep using calculate_optimal_timing
                self.current_green_time = max(self.current_green_time, MIN_GREEN_TIME)
                return

            # If target not currently green -> switch within AMBULANCE_SWITCH_DEADLINE
            if now - self.ambulance_requested_time >= AMBULANCE_SWITCH_DEADLINE:
                # perform immediate switch to ambulance lane
                if target_is_ns:
                    self.ns_green = True
                    self.sn_green = False
                else:
                    self.ns_green = False
                    self.sn_green = True
                self.signal_timer = now
                # set a modest green time for ambulance crossing (could be short)
                self.current_green_time = max(MIN_GREEN_TIME, 5.0)
                logger.info(f"Ambulance override: switched to {'N-S' if self.ns_green else 'S-N'} GREEN for ambulance")
            # else: wait until deadline before switching (so switch will happen within 3s)
            return

        # Normal dynamic operation if no ambulance priority
        ns_increased = (self.ns_vehicle_count - self.last_ns_vehicle_count) >= VEHICLE_THRESHOLD
        sn_increased = (self.sn_vehicle_count - self.last_sn_vehicle_count) >= VEHICLE_THRESHOLD

        # If sudden increase in either lane, recompute green time and restart timer in favor of current green side
        if ns_increased or sn_increased:
            self.current_green_time = self.calculate_optimal_timing()
            self.last_ns_vehicle_count = self.ns_vehicle_count
            self.last_sn_vehicle_count = self.sn_vehicle_count
            # restart timer to allow current green to handle sudden surge
            self.signal_timer = time.time()
            logger.info(f"Vehicle surge detected. Restarting timer: new green time = {self.current_green_time:.1f}s")
            return

        # Otherwise check usual expiration
        elapsed = time.time() - self.signal_timer
        if elapsed >= self.current_green_time:
            # Before toggling, check "keep-green-if-still-heavy" rule:
            # If current green lane still has significantly more vehicles than the other lane AND other lane < threshold -> keep green
            if self.ns_green:
                green_count = self.ns_vehicle_count
                red_count = self.sn_vehicle_count
                green_side = 'NS'
            else:
                green_count = self.sn_vehicle_count
                red_count = self.ns_vehicle_count
                green_side = 'SN'

            # If green side still has high demand and red side is below threshold => restart timer (keep green)
            if green_count > red_count and red_count < VEHICLE_THRESHOLD and green_count >= VEHICLE_THRESHOLD:
                # restart timer and extend green for the busy side
                self.current_green_time = self.calculate_optimal_timing()
                self.signal_timer = time.time()
                logger.info(f"Extending {green_side} GREEN because it still has higher demand ({green_count} vs {red_count})")
                return
            # normal toggle
            self.ns_green = not self.ns_green
            self.sn_green = not self.sn_green
            self.signal_timer = time.time()
            # compute new green time for new green side
            self.current_green_time = self.calculate_optimal_timing()
            logger.info(f"Signal Changed: {'N-S' if self.ns_green else 'S-N'} GREEN for {self.current_green_time:.1f}s")

    def draw_road_infrastructure(self):
        self.screen.fill(GRASS_COLOR)
        road_x = WIDTH // 2 - ROAD_WIDTH // 2
        pygame.draw.rect(self.screen, ROAD_COLOR, (road_x, 0, ROAD_WIDTH, HEIGHT))
        divider_x = WIDTH // 2
        for y in range(0, HEIGHT, 40):
            pygame.draw.rect(self.screen, YELLOW_COLOR, (divider_x - 2, y, 4, 20))
        pygame.draw.line(self.screen, LINE_COLOR, (road_x, 0), (road_x, HEIGHT), 3)
        pygame.draw.line(self.screen, LINE_COLOR, (road_x + ROAD_WIDTH, 0), (road_x + ROAD_WIDTH, HEIGHT), 3)

    def draw_traffic_lights(self):
        elapsed = time.time() - self.signal_timer
        remaining = max(0, self.current_green_time - elapsed)
        # N-S Light
        ns_light_x = WIDTH // 2 - 80
        ns_light_y = HEIGHT // 2 - 60
        pygame.draw.rect(self.screen, (0, 0, 0), (ns_light_x - 15, ns_light_y - 40, 30, 80))
        pygame.draw.circle(self.screen, (255, 0, 0) if not self.ns_green else (100, 0, 0), (ns_light_x, ns_light_y - 20), 10)
        pygame.draw.circle(self.screen, (0, 255, 0) if self.ns_green else (0, 100, 0), (ns_light_x, ns_light_y + 20), 10)
        label = self.font.render("N→S", True, (255, 255, 255))
        self.screen.blit(label, (ns_light_x - 20, ns_light_y + 50))
        if self.ns_green:
            timer_text = self.font.render(f"{remaining:.1f}s", True, (255, 255, 255))
            self.screen.blit(timer_text, (ns_light_x - 20, ns_light_y - 60))
        # S-N Light
        sn_light_x = WIDTH // 2 + 80
        sn_light_y = HEIGHT // 2 - 60
        pygame.draw.rect(self.screen, (0, 0, 0), (sn_light_x - 15, sn_light_y - 40, 30, 80))
        pygame.draw.circle(self.screen, (255, 0, 0) if not self.sn_green else (100, 0, 0), (sn_light_x, ns_light_y - 20), 10)
        pygame.draw.circle(self.screen, (0, 255, 0) if self.sn_green else (0, 100, 0), (sn_light_x, sn_light_y + 20), 10)
        label = self.font.render("S→N", True, (255, 255, 255))
        self.screen.blit(label, (sn_light_x - 20, sn_light_y + 50))
        if self.sn_green:
            timer_text = self.font.render(f"{remaining:.1f}s", True, (255, 255, 255))
            self.screen.blit(timer_text, (sn_light_x - 20, sn_light_y - 60))

    def draw_info_panel(self):
        panel_width = 350
        panel_height = 260
        info_rect = pygame.Rect(WIDTH - panel_width - 10, 10, panel_width, panel_height)
        # semi-transparent background is not trivial with pygame draw rect; use filled rect
        pygame.draw.rect(self.screen, (0, 0, 0), info_rect)
        pygame.draw.rect(self.screen, (255, 255, 255), info_rect, 2)
        title = self.large_font.render("Traffic Control Status", True, (255, 255, 255))
        self.screen.blit(title, (WIDTH - panel_width, 25))
        current_signal = "N→S GREEN" if self.ns_green else "S→N GREEN"
        signal_color = (0, 255, 0) if self.ns_green else (255, 100, 0)
        signal_text = self.font.render(f"Current: {current_signal}", True, signal_color)
        self.screen.blit(signal_text, (WIDTH - panel_width, 70))
        elapsed = time.time() - self.signal_timer
        remaining = max(0, self.current_green_time - elapsed)
        time_text = self.font.render(f"Time Remaining: {remaining:.1f}s", True, (255, 255, 255))
        self.screen.blit(time_text, (WIDTH - panel_width, 100))
        ns_text = f"N→S: Queue={self.ns_queue_length}, Vehicles={self.ns_vehicle_count}, Speed={self.ns_avg_speed:.1f}"
        sn_text = f"S→N: Queue={self.sn_queue_length}, Vehicles={self.sn_vehicle_count}, Speed={self.sn_avg_speed:.1f}"
        ns_color = (0, 255, 0) if self.ns_green else (255, 255, 255)
        sn_color = (0, 255, 0) if self.sn_green else (255, 255, 255)
        self.screen.blit(self.font.render(ns_text, True, ns_color), (WIDTH - panel_width, 140))
        self.screen.blit(self.font.render(sn_text, True, sn_color), (WIDTH - panel_width, 170))
        # Show demand
        ns_demand = self.ns_queue_length / max(0.1, self.ns_avg_speed)
        sn_demand = self.sn_queue_length / max(0.1, self.sn_avg_speed)
        self.screen.blit(self.font.render(f"Demand - N→S:{ns_demand:.1f}, S→N:{sn_demand:.1f}", True, (255, 255, 150)), (WIDTH - panel_width, 200))
        # Show 10% traffic reduction (example)
        reduction_factor = 0.9
        self.screen.blit(self.font.render(f"Traffic Reduced: N→S={self.ns_queue_length * (1 - reduction_factor):.1f}, S→N={self.sn_queue_length * (1 - reduction_factor):.1f}", True, (0, 255, 255)), (WIDTH - panel_width, 230))
        # Ambulance info
        amb_text = f"Ambulance Priority: {'ON' if self.ambulance_priority else 'OFF'}"
        self.screen.blit(self.font.render(amb_text, True, (255, 100, 100) if self.ambulance_priority else (200, 200, 200)), (WIDTH - panel_width, 255))

    def run_one_step(self):
        dt = self.clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return False
        self.update_signals()
        self.draw_road_infrastructure()
        for vehicle in self.ns_vehicles:
            vehicle.draw(self.screen)
        for vehicle in self.sn_vehicles:
            vehicle.draw(self.screen)
        self.draw_traffic_lights()
        self.draw_info_panel()
        pygame.display.flip()
        return self.running

    def close(self):
        pygame.quit()

# Camera processing for vehicle detection
class DirectionalCameraProcessor:
    def _init_(self, camera_id, direction_name, ambulance_model_path="ambulance.pt"):
        self.camera_id = camera_id
        self.direction_name = direction_name
        # General detector: keep existing yolov8 (detect vehicles only; people excluded)
        self.model = YOLO("yolov8n.pt")
        # Vehicle classes to detect in general model: use COCO ids -> car=2, motorcycle=3, bus=5, truck=7 (exclude person 0)
        self.vehicle_classes = [2, 3, 5, 7]
        self.tracker = VehicleTracker()
        self.last_data = {'queue_length': 0, 'avg_speed': 1.0, 'vehicle_count': 0}
        # Ambulance model (trained on your ambulance dataset). If missing, ambulance detection will be skipped with warning.
        try:
            self.ambulance_model = YOLO("C:/Users/srian/OneDrive/Desktop/clg/SIH25/best.pt")
            logger.info(f"Loaded ambulance model from {ambulance_model_path}")
        except Exception as e:
            logger.warning(f"Could not load ambulance model '{ambulance_model_path}': {e}. Ambulance detection disabled for {direction_name}")
            self.ambulance_model = None
        self.ambulance_detected = False
        self.ambulance_last_seen = 0.0
        # detection thresholds
        self.general_conf_threshold = 0.2
        self.ambulance_conf_threshold = 0.5  # Lowered to detect the ambulance; adjust based on logs if needed

    def process_frame(self, frame):
        if frame is None:
            return self.last_data

        detections = []
        # 1) Ambulance detection first (priority)
        self.ambulance_detected = False
        if self.ambulance_model is not None:
            try:
                amb_results = self.ambulance_model(frame)
                for r in amb_results:
                    if hasattr(r, 'boxes') and r.boxes is not None:
                        for box in r.boxes:
                            conf = float(box.conf[0]) if hasattr(box, 'conf') else 0.0
                            cls = int(box.cls[0]) if hasattr(box, 'cls') else 0  # Assume class 0 is ambulance if custom-trained
                            coords = box.xyxy[0] if hasattr(box, 'xyxy') else None
                            logger.info(f"Ambulance model in {self.direction_name}: detected class {cls} with conf {conf:.2f}")
                            # Only consider if it's the ambulance class (for custom models) and high confidence
                            if cls == 0 and coords is not None and conf >= self.ambulance_conf_threshold:
                                x1, y1, x2, y2 = map(int, coords)
                                # draw red box for ambulance
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                                cv2.putText(frame, f'AMBULANCE {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                                self.ambulance_detected = True
                                self.ambulance_last_seen = time.time()
            except Exception as e:
                logger.exception(f"Ambulance model inference error: {e}")

        # 2) General vehicle detection (exclude persons)
        try:
            results = self.model(frame, stream=False)
            for r in results:
                if hasattr(r, 'boxes') and r.boxes is not None:
                    for box in r.boxes:
                        cls = int(box.cls[0]) if hasattr(box, 'cls') else None
                        conf = float(box.conf[0]) if hasattr(box, 'conf') else 0.0
                        coords = box.xyxy[0] if hasattr(box, 'xyxy') else None
                        if cls in self.vehicle_classes and coords is not None and conf > self.general_conf_threshold:
                            x1, y1, x2, y2 = map(int, coords)
                            detections.append((x1, y1, x2, y2))
                            # green boxes for normal vehicles
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame, f'Vehicle {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        except Exception as e:
            logger.exception(f"General model inference error: {e}")

        # track and compute simple stats
        tracked = self.tracker.update(detections)
        queue_length = len(detections)
        speeds = [v['speed'] for v in tracked if v['speed'] > 0]
        avg_speed = np.mean(speeds) if speeds else 1.0
        self.last_data = {'queue_length': queue_length, 'avg_speed': max(0.1, avg_speed), 'vehicle_count': len(tracked)}
        # annotate camera frame
        cv2.putText(frame, f'{self.direction_name} Direction', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.putText(frame, f'Queue:{queue_length}, Speed:{avg_speed:.1f}, Tracked:{len(tracked)}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # if ambulance recently seen, keep ambulance_detected true for a short hold window
        if not self.ambulance_detected and (time.time() - self.ambulance_last_seen) < 1.0:
            self.ambulance_detected = True

        return self.last_data

# Main system
class Smart2WayTrafficSystem:
    def _init_(self):
        self.simulator = TwoWayTrafficSimulator()
        # processors: left camera -> N→S, right camera -> S→N
        self.ns_processor = DirectionalCameraProcessor(0, "N→S")
        self.sn_processor = DirectionalCameraProcessor(1, "S→N")
        # video captures
        self.cap_ns = cv2.VideoCapture(0)
        self.cap_sn = cv2.VideoCapture(1)
        for cap in [self.cap_ns, self.cap_sn]:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
        logger.info("Smart 2-Way Traffic System Initialized")

    def run(self):
        logger.info("Starting Smart 2-Way Traffic Management System...")
        try:
            while self.simulator.running:
                ret_ns, frame_ns = self.cap_ns.read()
                ret_sn, frame_sn = self.cap_sn.read()

                if not ret_ns or not ret_sn:
                    # fallback dummy stats if camera not available
                    ns_data = {'queue_length': 2, 'avg_speed': 1.0, 'vehicle_count': 2}
                    sn_data = {'queue_length': 1, 'avg_speed': 1.5, 'vehicle_count': 1}
                    # no ambulance info available
                    ns_amb = False
                    sn_amb = False
                else:
                    ns_data = self.ns_processor.process_frame(frame_ns)
                    sn_data = self.sn_processor.process_frame(frame_sn)
                    ns_amb = self.ns_processor.ambulance_detected
                    sn_amb = self.sn_processor.ambulance_detected

                # If any ambulance detected, request priority
                if ns_amb:
                    self.simulator.request_ambulance_priority('NS')
                elif sn_amb:
                    self.simulator.request_ambulance_priority('SN')
                else:
                    # If previously had an ambulance priority and now none detected for both, clear priority
                    if self.simulator.ambulance_priority:
                        # small hold: ensure both processors have no ambulance recently
                        if (time.time() - self.ns_processor.ambulance_last_seen > 1.5) and (time.time() - self.sn_processor.ambulance_last_seen > 1.5):
                            self.simulator.clear_ambulance_priority()

                # push data to simulator (it will handle timing including ambulance priority)
                self.simulator.update_traffic_data(ns_data, sn_data)

                # run one step of display/timing
                if not self.simulator.run_one_step():
                    break

                # show camera frames combined
                if ret_ns and ret_sn:
                    try:
                        display_ns = cv2.resize(frame_ns, (400, 300))
                        display_sn = cv2.resize(frame_sn, (400, 300))
                        combined_frame = np.hstack([display_ns, display_sn])
                        cv2.imshow('Traffic Cameras - N→S (Left) | S→N (Right)', combined_frame)
                    except Exception as e:
                        logger.warning(f"Could not display camera frames: {e}")

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self.cleanup()

    def cleanup(self):
        logger.info("Shutting down Smart Traffic System...")
        try:
            self.cap_ns.release()
            self.cap_sn.release()
        except Exception:
            pass
        cv2.destroyAllWindows()
        self.simulator.close()
        logger.info("System stopped successfully.")

# Main execution
if _name_ == "_main_":
    try:
        system = Smart2WayTrafficSystem()
        system.run()
    except Exception as e:
        logger.exception(f"Fatal error running system: {e}")
    finally:
        cv2.destroyAllWindows()
        pygame.quit()
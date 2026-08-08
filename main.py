import cv2
import time
import argparse
import os
import numpy as np
import logging
from datetime import datetime
from ultralytics import YOLO

# ==========================================
# 1. ตั้งค่าระบบ Logging
# ==========================================
# ระบบจะเขียน Log ทั้งลงหน้าจอ Terminal และลงไฟล์ cctv_system.log
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler("cctv_system.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)

def update_heartbeat():
    """ฟังก์ชันอัปเดตเวลาล่าสุดที่โปรแกรมยังมีชีวิตอยู่"""
    try:
        with open("last_alive.txt", "w", encoding='utf-8') as f:
            f.write(f"System was last running at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("If the time is not updating, the program has crashed or stopped.")
    except Exception as e:
        logging.error(f"Cannot write heartbeat: {e}")

class ChairTracker:
    def __init__(self, chair_id, roi_coords):
        self.chair_id = chair_id
        self.roi = roi_coords
        
        self.state = "EMPTY"
        self.sit_start = 0
        self.empty_start = 0
        self.count = 0
        self.counted = False
        
        # Eigenface Variables
        self.recognizer = cv2.face.EigenFaceRecognizer_create()
        self.face_samples = []
        self.face_labels = []
        self.is_model_trained = False
        self.collection_start_time = 0
        self.last_capture_time = 0
        self.collection_done = False
        self.occupied_start = 0

    def get_face_crop(self, frame, box):
        """ตัดภาพเฉพาะส่วนหัวและแปลงเป็นขาวดำขนาด 100x100 สำหรับ Eigenface"""
        bx1, by1, bx2, by2 = map(int, box)
        h, w = frame.shape[:2]
        bx1, by1 = max(0, bx1), max(0, by1)
        bx2, by2 = min(w, bx2), min(h, by2)
        
        if bx2 <= bx1 or by2 <= by1:
            return None
            
        crop = frame[by1:by2, bx1:bx2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (100, 100))
        return resized

    def update(self, frame, detections):
        current_time = time.time()
        
        # เช็คว่ามีคนอยู่ใน ROI ของเก้าอี้นี้ไหม
        head_detected = False
        best_box = None
        for box in detections:
            bx1, by1, bx2, by2 = box
            center_x = (bx1 + bx2) / 2
            center_y = (by1 + by2) / 2
            rx1, ry1, rx2, ry2 = self.roi
            if (rx1 <= center_x <= rx2) and (ry1 <= center_y <= ry2):
                head_detected = True
                best_box = box
                break

        # ==========================================
        # STATE MACHINE LOGIC (V2 - Eigenface)
        # ==========================================
        if self.state == "EMPTY":
            if head_detected:
                logging.info(f"[Chair {self.chair_id}] EMPTY -> SITTING (Motion detected)")
                self.state = "SITTING"
                self.sit_start = current_time

        elif self.state == "SITTING":
            if head_detected:
                if current_time - self.sit_start >= 5.0: # นั่งต่อเนื่อง 5 วินาที
                    if not self.counted:
                        is_new_customer = True
                        
                        # เช็คหน้าเดิมด้วย Eigenface (ถ้าโมเดลถูกเทรนแล้ว)
                        if self.is_model_trained and best_box is not None:
                            face_crop = self.get_face_crop(frame, best_box)
                            if face_crop is not None:
                                label, confidence = self.recognizer.predict(face_crop)
                                logging.info(f"[Chair {self.chair_id}] AI FACE MATCH CONFIDENCE: {confidence:.2f}")
                                
                                # **จุดจูนความแม่นยำ:** ค่าความมั่นใจต่ำ = หน้าเหมือนเดิมมาก
                                if confidence < 5000: 
                                    logging.info(f"[Chair {self.chair_id}] SAME CUSTOMER RETURNING. Ignoring False Count.")
                                    is_new_customer = False
                                else:
                                    logging.info(f"[Chair {self.chair_id}] NEW FACE DETECTED.")

                        if is_new_customer:
                            self.count += 1
                            logging.info(f"[Chair {self.chair_id}] COUNT +1! (Total: {self.count})")
                            
                            # ถ่ายภาพหลักฐานตอนนับ
                            filename = f"captures/Chair{self.chair_id}_Count{self.count}_{int(current_time)}.jpg"
                            cv2.imwrite(filename, frame)
                            logging.info(f"[Chair {self.chair_id}] Snapshot saved: {filename}")
                            
                            # รีเซ็ตข้อมูล AI เพื่อเตรียมเก็บหน้าลูกค้าคนใหม่
                            self.face_samples = []
                            self.face_labels = []
                            self.is_model_trained = False
                            self.collection_done = False
                        
                        self.counted = True
                        
                    self.state = "OCCUPIED"
                    self.occupied_start = current_time
            else:
                self.state = "LEAVING"
                self.empty_start = current_time

        elif self.state == "OCCUPIED":
            if not head_detected:
                self.state = "LEAVING"
                self.empty_start = current_time
            else:
                # ระบบเก็บภาพเทรน AI (เก็บ 30 ภาพ, ภาพละ 3 วิ, เริ่มเก็บหลังลูกค้านั่งครบ 1 นาที)
                if not self.collection_done and best_box is not None:
                    if current_time - self.occupied_start >= 60.0: 
                        if current_time - self.last_capture_time >= 3.0: 
                            face_crop = self.get_face_crop(frame, best_box)
                            if face_crop is not None:
                                self.face_samples.append(face_crop)
                                self.face_labels.append(1)
                                self.last_capture_time = current_time
                                logging.debug(f"[Chair {self.chair_id}] Collected training sample {len(self.face_samples)}/30")
                                
                                if len(self.face_samples) >= 30:
                                    logging.info(f"[Chair {self.chair_id}] Training EigenFace Model with 30 samples...")
                                    self.recognizer.train(self.face_samples, np.array(self.face_labels))
                                    self.is_model_trained = True
                                    self.collection_done = True
                                    logging.info(f"[Chair {self.chair_id}] Training Complete. Ready for verification.")

        elif self.state == "LEAVING":
            if head_detected:
                if self.counted:
                    self.state = "OCCUPIED"
                else:
                    self.state = "SITTING"
            else:
                if current_time - self.empty_start >= 5.0:
                    self.state = "EMPTY"
                    if self.counted:
                        logging.info(f"[Chair {self.chair_id}] Customer left completely. Ready for next.")
                    self.counted = False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="RTSP URL")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--headless", action="store_true", help="Run without UI for production")
    args = parser.parse_args()

    # สร้างโฟลเดอร์สำหรับเก็บรูป
    if not os.path.exists("captures"):
        os.makedirs("captures")

    logging.info("Starting CCTV Customer Counting System (V2: Eigenface Edition)")
    logging.info("Loading YOLO model...")
    model = YOLO('yolov8n.pt') 

    chair1_roi = [302, 259, 638, 606]
    chair2_roi = [714, 184, 981, 500]

    chairs = [
        ChairTracker(chair_id=1, roi_coords=chair1_roi),
        ChairTracker(chair_id=2, roi_coords=chair2_roi)
    ]

    process_interval = 0.33  
    last_process_time = time.time()
    last_heartbeat_time = time.time()
    latest_detections = []

    # ลูปวงนอก เอาไว้ Reconnect ถ่ายทอดสดหลุด
    while True:
        logging.info(f"Connecting to video stream: {args.video}")
        cap = cv2.VideoCapture(args.video)
        
        if not cap.isOpened():
            logging.error("Cannot open video stream. Retrying in 5 seconds...")
            time.sleep(5)
            continue
            
        logging.info("Stream Connected Successfully.")

        while True:
            ret, frame = cap.read()
            if not ret:
                logging.warning("Stream disconnected or frame error. Attempting to reconnect...")
                break

            frame = cv2.resize(frame, (args.width, args.height))
            current_time = time.time()

            # 1. เขียน Heartbeat ทุกๆ 10 วินาที
            if current_time - last_heartbeat_time >= 10.0:
                update_heartbeat()
                last_heartbeat_time = current_time

            # 2. ประมวลผล AI ทุกๆ 0.33 วินาที (3 FPS)
            if current_time - last_process_time >= process_interval:
                last_process_time = current_time
                results = model(frame, verbose=False, classes=[0]) 
                latest_detections = results[0].boxes.xyxy.cpu().numpy()

                for chair in chairs:
                    chair.update(frame, latest_detections)

            # 3. วาด UI (ถ้าไม่ได้รันโหมด Headless)
            if not args.headless:
                for chair in chairs:
                    x1, y1, x2, y2 = chair.roi
                    color = (0, 255, 0) if chair.state in ["SITTING", "OCCUPIED"] else (0, 0, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    status_text = f"C{chair.chair_id} | State: {chair.state} | Count: {chair.count}"
                    cv2.putText(frame, status_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                cv2.imshow("CCTV Eigenface System", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    logging.info("System manually shut down by user.")
                    return

        cap.release()
        time.sleep(2) # รอแป๊บนึงก่อนพยายามต่อใหม่

if __name__ == "__main__":
    main()

from ultralytics import YOLO
import cv2
from pathlib import Path
import time
from datetime import datetime

class GoldDetectorROI:
    def __init__(self, model_path="best.pt", webcam_index=0, roi=(200, 100, 500, 400)):
        """
        Initialize the detector.
        :param model_path: path to YOLO weights
        :param webcam_index: webcam ID
        :param roi: tuple of (x1, y1, x2, y2) defining the region of interest
        """
        self.model = YOLO(model_path)
        self.cap = cv2.VideoCapture(webcam_index)
        self.roi = roi  # ROI as (x1, y1, x2, y2)
        
        #Recording attributes
        self.recording = False
        self.writer = None
        self.last_detection_time = 0
        self.out_file = None
        self.out_dir = Path("runs/recordings")
        self.out_dir.mkdir(parents = True, exist_ok = True)
        self.fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        

    def crop_roi(self, frame):
        """Crop the ROI from the frame."""
        x1, y1, x2, y2 = self.roi
        return frame[y1:y2, x1:x2] #returns the cropped frame

    def convert_to_full_coords(self, box):
        """Convert ROI-relative box coordinates to full frame coordinates.""" 
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        roi_x1, roi_y1, _, _ = self.roi
        x1 += roi_x1
        x2 += roi_x1
        y1 += roi_y1
        y2 += roi_y1
        return x1, y1, x2, y2

    def draw_roi_box(self, frame):
        """Draw the ROI rectangle on the frame."""
        x1, y1, x2, y2 = self.roi
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    def draw_detection(self, frame, coords, conf):
        """Draw detection box and label on frame."""
        x1, y1, x2, y2 = coords
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"Gold Detected {conf:.2f}"
        cv2.putText(frame, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    def run(self):
        """Main loop for webcam capture, detection, display and recording."""
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Draw ROI
            self.draw_roi_box(frame)

            # Crop ROI for detection
            roi_frame = self.crop_roi(frame)

            # Run YOLO only on ROI
            results = self.model(roi_frame, conf=0.2)[0]

            gold_detected = False
            
            # Process detections
            for box in results.boxes:
                coords = self.convert_to_full_coords(box)
                conf = box.conf.item()
                self.draw_detection(frame, coords, conf)
                gold_detected = True
            
            #Handle recording
            current_time = time.time()
            if gold_detected:
                self.last_detection_time = current_time
                if not self.recording:
                    h,w = frame.shape[:2] #takes the 2 elements height and width not the channels
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    self.out_file = self.out_dir / f"gold_detected_{ts}.mp4"
                    self.writer = cv2.VideoWriter(str(self.out_file), self.fourcc, 20.0, (w,h))
                    self.recording = True
                    print(f"Recording started: {self.out_file}")
           
            #Closing the recorder
            if self.recording  and (current_time - self.last_detection_time) > 30:
                self.writer.release()
                self.writer = None
                self.recording = False                
                print(f"Recording stopped: {self.out_file}")
            
            
            #like taking the screen recording 
            if self.recording and self.writer is not None:
                self.writer.write(frame)
                
            # Display
            cv2.imshow("Gold Detection (ROI)", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    detector = GoldDetectorROI(model_path="best.pt", roi=(200, 100, 500, 400))
    detector.run()

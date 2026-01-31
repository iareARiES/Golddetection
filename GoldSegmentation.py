from torch import classes
from ultralytics import YOLO
import cv2
from pathlib import Path
import time
from datetime import datetime




# ------------------- YOLO26 SEGMENTATION -------------------
class YOLOSegmentation:
    def __init__(self, model_path="yolo26n-seg.pt", roi=None, classes=[0]):
        """
        Run YOLO26 segmentation on ROI.
        classes=[0] means 'person' only.
        """
        self.model = YOLO(model_path)
        self.roi = roi
        self.classes = classes
        
        
    def crop_roi(self,frame):
        x1,y1,x2,y2 = self.roi
        return frame[y1:y2, x1:x2]
        
    def convert_to_full_coords(self,box):
        x1,y1,x2,y2 = map(int, box.xyxy[0])
        roi_x1,roi_y1,_,_ = self.roi
        x1 += roi_x1
        x2 += roi_x1
        y1 += roi_y1
        y2 += roi_y1
        return x1,y1,x2,y2
        
    def draw_detection(self,frame, coords, conf):
        x1,y1,x2,y2 = coords
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        label = f"Person {conf:2f}"
        cv2.putText(frame, label, (x1,y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255),2)
            
    def predict_and_draw(self, frame):
        roi_frame = self.crop_roi(frame)
        results = self.model(roi_frame, classes=self.classes, conf=0.2)[0]
        
        # This draws the **segmentation masks on the ROI**
        annotated_roi = results.plot()  # <- this handles pixelated masks
        
        # Put the annotated ROI back into the original frame
        x1, y1, x2, y2 = self.roi
        frame[y1:y2, x1:x2] = annotated_roi
        
        return frame



# ------------------- GOLD DETECTOR LAYER-------------------
class GoldDetectorROI:
    def __init__(self, model_path="best.pt", webcam_index=0, roi=None):
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


    def draw_detection(self, frame, coords, conf):
        """Draw detection box and label on frame."""
        x1, y1, x2, y2 = coords
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"Gold Detected {conf:.2f}"
        cv2.putText(frame, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    
    def predict_and_draw(self, frame):
        """
        Run gold detection on ROI, draw results on frame,
        and handle recording logic. Returns frame + gold_detected flag.
        """
        roi_frame = self.crop_roi(frame)
        results = self.model(roi_frame, conf = 0.2)[0]
        
        gold_detected = False
        for box in results.boxes:
            coords = self.convert_to_full_coords(box)
            conf = box.conf.item()
            self.draw_detection(frame,coords,conf)
            gold_detected = True
        
        return frame, gold_detected
    
    
    def handle_recording(self, frame, gold_detected):
        """
        Start/stop recording if gold is detected.
        """
        current_time = time.time()
        if gold_detected:
            self.last_detection_time = current_time
            if not self.recording:
                h, w = frame.shape[:2]
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                self.out_file = self.out_dir / f"gold_detected_{ts}.mp4"
                self.writer = cv2.VideoWriter(str(self.out_file), self.fourcc, 20.0, (w, h))
                self.recording = True
                print(f"Recording started: {self.out_file}")

        if self.recording and (current_time - self.last_detection_time) > 30:
            self.writer.release()
            self.writer = None
            self.recording = False
            print(f"Recording stopped: {self.out_file}")

        if self.recording and self.writer is not None:
            self.writer.write(frame)    
        


class MultiDetectorROI:
    def __init__(self, gold_model_path="best.pt", yolo26_model_path="yolo26n-seg.pt", roi=None):
        self.cap = cv2.VideoCapture(0)
        self.roi = roi
        self.gold_detector = GoldDetectorROI(
            model_path=gold_model_path,
            webcam_index=0,      # must be int
            roi=roi
        )
        self.seg_detector = YOLOSegmentation(yolo26_model_path, roi)


    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Draw ROI rectangle
            x1, y1, x2, y2 = self.roi
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # GOLD DETECTION + RECORDING
            frame, gold_detected = self.gold_detector.predict_and_draw(frame)
            self.gold_detector.handle_recording(frame, gold_detected)

            # YOLO26 SEGMENTATION
            frame = self.seg_detector.predict_and_draw(frame)

            cv2.imshow("Gold + YOLO26 Detection (ROI)", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        self.cap.release()
        cv2.destroyAllWindows()

# ------------------- RUN -------------------
if __name__ == "__main__":
    seg_x1 = 200
    seg_y1 = 200 
    seg_x2 = 800
    seg_y2 = 800
    
    detector = MultiDetectorROI(
        gold_model_path="best.pt",
        yolo26_model_path="yolo26n-seg.pt",
        roi=(seg_x1,seg_y1,seg_x2, seg_y2)
    )
    detector.run()
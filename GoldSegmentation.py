from ultralytics import YOLO
import cv2
from pathlib import Path
import time
from datetime import datetime
import numpy as np
from paddleocr import PaddleOCR


def get_screen_resolution():
    """
    Auto-detect screen resolution. Works on Windows, Linux, and Raspberry Pi.
    Returns (width, height) or None if detection fails.
    """
    # Method 1: Try tkinter (works on most systems)
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()  # Hide the window
        width = root.winfo_screenwidth()
        height = root.winfo_screenheight()
        root.destroy()
        return (width, height)
    except:
        pass
    
    # Method 2: Try xrandr on Linux/RPi
    try:
        import subprocess
        output = subprocess.check_output(['xrandr']).decode('utf-8')
        for line in output.split('\n'):
            if '*' in line:  # Current resolution has asterisk
                resolution = line.split()[0]
                width, height = map(int, resolution.split('x'))
                return (width, height)
    except:
        pass
    
    # Method 3: Fallback - return None to use camera resolution
    return None


def resize_for_display(frame, display_width, display_height):
    """Resize frame to fit display dimensions while maintaining aspect ratio."""
    h, w = frame.shape[:2]
    scale_w = display_width / w
    scale_h = display_height / h
    scale = min(scale_w, scale_h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


# ------------------- YOLO26 SEGMENTATION -------------------
class YOLOSegmentation:
    def __init__(self, model_path=None, roi=None, classes=[0]):
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
        label = f"Person {conf:.2f}"
        cv2.putText(frame, label, (x1,y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255),2)
            
    def predict_and_draw(self, frame, results):
        """
        Draw segmentation results directly on the frame at correct ROI-offset positions.
        No resizing or pasting - draws on original frame.
        """
        x1_roi, y1_roi, x2_roi, y2_roi = map(int, self.roi)
        
        # Draw masks if available
        if results.masks is not None:
            for mask_points in results.masks.xy:
                # Offset mask points by ROI position
                contour = np.array(mask_points, dtype=np.int32)
                contour[:, 0] += x1_roi  # Offset X
                contour[:, 1] += y1_roi  # Offset Y
                
                # Draw filled mask with transparency
                overlay = frame.copy()
                cv2.fillPoly(overlay, [contour], (0, 255, 0))
                cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
                
                # Draw mask outline
                cv2.polylines(frame, [contour], True, (0, 255, 0), 2)
        
        # Draw boxes if available
        if results.boxes is not None:
            for box in results.boxes:
                bx1, by1, bx2, by2 = map(int, box.xyxy[0])
                # Offset by ROI position
                bx1 += x1_roi
                bx2 += x1_roi
                by1 += y1_roi
                by2 += y1_roi
                
                conf = box.conf.item()
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
                label = f"Person {conf:.2f}"
                cv2.putText(frame, label, (bx1, by1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        return frame







# ------------------- GOLD DETECTOR LAYER-------------------
class GoldDetectorROI:
    def __init__(self, model_path=None, roi=None):
        """
        Initialize the detector.
        :param model_path: path to YOLO weights
        :param webcam_index: webcam ID
        :param roi: tuple of (x1, y1, x2, y2) defining the region of interest
        """
        self.model = YOLO(model_path)
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
            if self.writer is not None:
                self.writer.release()
            self.writer = None
            self.recording = False

        if self.recording and self.writer is not None:
            self.writer.write(frame)    
        


class MultiDetectorROI:
    def __init__(self, gold_model_path=None, yolo26_model_path=None, roi=None, screen_resolution=None):
        self.cap = cv2.VideoCapture(0)
        self.roi = roi
        self.gold_detector = GoldDetectorROI(
            model_path=gold_model_path,
            roi=roi
        )
        self.seg_detector = YOLOSegmentation(yolo26_model_path, roi)
        # Screen resolution for display scaling (width, height)
        # If None, will auto-detect from first frame
        self.screen_resolution = screen_resolution

    #overlap solution
    def box_overlaps_mask(self, box_coords, person_masks, roi):
        """
        True overlap check using binary mask intersection.
        """
        if person_masks is None:
            return False

        x1, y1, x2, y2 = box_coords
        roi_x1, roi_y1, roi_x2, roi_y2 = roi

        roi_w = roi_x2 - roi_x1
        roi_h = roi_y2 - roi_y1

        # Create empty person mask image
        person_binary = np.zeros((roi_h, roi_w), dtype=np.uint8)

        # Draw each person contour
        for mask_points in person_masks:
            contour = np.array(mask_points, dtype=np.int32)
            cv2.fillPoly(person_binary, [contour], 255)

        # Create gold box mask (ROI-relative)
        box_mask = np.zeros((roi_h, roi_w), dtype=np.uint8)

        bx1 = x1 - roi_x1
        by1 = y1 - roi_y1
        bx2 = x2 - roi_x1
        by2 = y2 - roi_y1

        cv2.rectangle(box_mask, (bx1, by1), (bx2, by2), 255, -1)

        # Intersection
        overlap = cv2.bitwise_and(person_binary, box_mask)

        # If any overlap pixels exist → reject
        return np.any(overlap)


        
    def resize_with_aspect_ratio(self, frame, target_width, target_height):
        """
        Resize frame to fit within target dimensions while maintaining aspect ratio.
        """
        h, w = frame.shape[:2]
        
        # Calculate scaling factor to fit within target dimensions
        scale_w = target_width / w
        scale_h = target_height / h
        scale = min(scale_w, scale_h)  # Use smaller scale to fit both dimensions
        
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        return resized
    
    def draw_status_panel(self, frame, gold_detected):
        """
        Draw status labels on the right side of the frame.
        Shows: Gold Detected (Yes/No), Recording (Yes/No), Recording Duration
        """
        h, w = frame.shape[:2]
        
        # Panel settings
        panel_width = 200
        panel_x = w - panel_width - 10
        start_y = 30
        line_height = 35
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        
        # Gold Detection Status
        gold_text = "Gold: YES" if gold_detected else "Gold: NO"
        gold_color = (0, 255, 0) if gold_detected else (0, 0, 255)
        cv2.putText(frame, gold_text, (panel_x, start_y), font, font_scale, gold_color, thickness)
        
        # Recording Status
        is_recording = self.gold_detector.recording
        rec_text = "Recording: YES" if is_recording else "Recording: NO"
        rec_color = (0, 255, 0) if is_recording else (0, 0, 255)
        cv2.putText(frame, rec_text, (panel_x, start_y + line_height), font, font_scale, rec_color, thickness)
        
        # Recording Duration
        if is_recording and self.recording_start_time is not None:
            duration = time.time() - self.recording_start_time
            minutes = int(duration // 60)
            seconds = int(duration % 60)
            duration_text = f"Duration: {minutes:02d}:{seconds:02d}"
        else:
            duration_text = "Duration: 00:00"
        cv2.putText(frame, duration_text, (panel_x, start_y + 2 * line_height), font, font_scale, (255, 255, 255), thickness)
        
        return frame
    
    def run(self):
        # Track recording start time
        self.recording_start_time = None
        
        # Read first frame to get camera dimensions
        ret, first_frame = self.cap.read()
        if not ret:
            print("Error: Could not read from camera")
            return
        
        cam_height, cam_width = first_frame.shape[:2]
        print(f"Camera resolution: {cam_width}x{cam_height}")
        
        # ---------------- OpenCV window setup ----------------
        window_name = "Gold + YOLO26 Detection (ROI)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # allow resizing
        
        # Auto-detect screen resolution
        screen_res = get_screen_resolution()
        if screen_res:
            screen_width, screen_height = screen_res
            print(f"Auto-detected screen: {screen_width}x{screen_height}")
        else:
            # fallback to camera frame size
            screen_width, screen_height = cam_width, cam_height
            print(f"Using camera resolution: {screen_width}x{screen_height}")
        
        # Calculate scale to fit screen (preserve aspect ratio)
        scale_w = screen_width / cam_width
        scale_h = screen_height / cam_height
        scale = min(scale_w, scale_h)
        
        display_width = int(cam_width * scale)
        display_height = int(cam_height * scale)
        print(f"Display size: {display_width}x{display_height}")
        
        # Set OpenCV window size
        cv2.resizeWindow(window_name, display_width, display_height)
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Draw ROI rectangle
            x1, y1, x2, y2 = self.roi
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # ---- SEGMENTATION ONCE ----
            roi_frame = self.seg_detector.crop_roi(frame)
            seg_results = self.seg_detector.model(
                roi_frame,
                classes=self.seg_detector.classes,
                conf=0.2
            )[0]

            person_masks = seg_results.masks.xy if seg_results.masks else None

            # Draw segmentation
            frame = self.seg_detector.predict_and_draw(frame, seg_results)

            # ---- GOLD FILTERING ----
            frame, gold_detected = self.detect_gold_filtered(frame, person_masks)
            
            # Track recording start time
            was_recording = self.gold_detector.recording
            self.gold_detector.handle_recording(frame, gold_detected)
            
            # Update recording start time
            if self.gold_detector.recording and not was_recording:
                self.recording_start_time = time.time()
            elif not self.gold_detector.recording:
                self.recording_start_time = None

            # Draw status panel on the right
            frame = self.draw_status_panel(frame, gold_detected)
            
            # Resize and display
            display_frame = resize_for_display(frame, display_width, display_height)
            cv2.imshow(window_name, display_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        if self.gold_detector.writer is not None:
            self.gold_detector.writer.release()
            self.gold_detector.writer = None
            self.gold_detector.recording = False
            print("Recording safely closed.")

        self.cap.release()
        cv2.destroyAllWindows()


    def detect_gold_filtered(self, frame, person_masks):
        """
        Run gold detection but filter out detections that overlap with person masks.
        """
        roi_frame = self.gold_detector.crop_roi(frame)
        results = self.gold_detector.model(roi_frame, conf=0.2)[0]
        
        gold_detected = False
        for box in results.boxes:
            coords = self.gold_detector.convert_to_full_coords(box)
            
            # Check if this gold detection overlaps with any person
            if self.box_overlaps_mask(coords, person_masks, self.roi):
                continue  # Skip completely, draw nothing

            
            # Valid gold detection (not on person)
            conf = box.conf.item()
            self.gold_detector.draw_detection(frame, coords, conf)
            gold_detected = True
    
        return frame, gold_detected

# ------------------- RUN -------------------
if __name__ == "__main__":
    seg_x1 = 200
    seg_y1 = 200 
    seg_x2 = 800
    seg_y2 = 800
    
    # Screen resolution - set to None for auto-detection (recommended)
    # Or manually set, e.g., (800, 480) for RPi 7" screen
    SCREEN_RESOLUTION = None  # Auto-detect
    
    detector = MultiDetectorROI(
        gold_model_path="best.onnx",
        yolo26_model_path="yolo26n-seg.onnx",
        roi=(seg_x1,seg_y1,seg_x2, seg_y2),
        screen_resolution=SCREEN_RESOLUTION
    )
    detector.run()
 

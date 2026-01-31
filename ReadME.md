class GoldDetectorROI:
    # Handles gold detection + recording

class YOLO26SegROI:
    # Handles person/other class segmentation

class MultiDetectorROI:
    # Orchestrates both classes
    # Reads frame once, passes cropped ROI to both detectors
    # Combines drawings on the same frame
from ultralytics import YOLO
import time
# Load a model
model = YOLO("best.pt")

# Export the model to engine format
model.export(format="engine", dynamic=True, verbose=False, half=True, int8=True) # Creates yolov8n.engine


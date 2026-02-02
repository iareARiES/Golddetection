from ultralytics import YOLO

model1 = YOLO("best.pt")
print("=== BEST.PT ===")
print("Task:", model1.task)
print("Model:", model1.model)

model2 = YOLO("yolo26n-seg.pt")
print("\n=== YOLO26 SEG ===")
print("Task:", model2.task)
print("Model:", model2.model)

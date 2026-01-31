from ultralytics import YOLO
import cv2

model = YOLO("yolo26n-seg.pt")

cap = cv2.VideoCapture(0)

while True:
    ret,frame = cap.read()
    if not ret:
        break

    results = model(frame, classes=[0])
    
    annotated_frame = results[0].plot()
    
    cv2.imshow("YOLO SEGMENTATION", annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    
cap.release()
cv2.destroyAllWindows()
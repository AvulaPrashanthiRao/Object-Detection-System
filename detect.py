from ultralytics import YOLO
import cv2

# Load model
model = YOLO("yolov8s.pt")

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection
    results = model(frame, conf=0.5)

    # Draw boxes
    for r in results:
        annotated_frame = r.plot()

    # Show frame
    cv2.imshow("Webcam Detection", annotated_frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
import cv2
from ultralytics import YOLO

model = YOLO('best.pt')

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()

    if success:
        results = model.predict(frame, verbose=False)

        annotated_frame = results[0].plot()
        for x in results[0].boxes:
            if x.conf.item() < 0.30:
                cv2.imwrite(f"add_to_dataset/im_{x.conf.item()}.jpg", frame)
        cv2.imshow("YOLOv8 Tracking", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
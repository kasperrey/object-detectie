import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('best.pt')

# Open the video file
result = model.predict("pentago.jpg")
print(result[0].boxes[0].conf.item() > 0.58)
print(result[0].boxes[0].conf)
cv2.imshow("test", result[0].plot())
result = model.predict("pentago.jpeg")
print(result[0].boxes[0].conf)
cv2.imshow("te", result[0].plot())
cv2.waitKey(0)
cv2.destroyAllWindows()


from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('best.pt')

results = model.train(data='spellen.yaml', epochs=50, batch=-1)

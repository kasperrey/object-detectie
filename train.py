from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('dataset2/best.pt')

results = model.train(data='spellen2.yaml', epochs=100, batch=-1)

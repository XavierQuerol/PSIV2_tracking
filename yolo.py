from ultralytics import YOLO

# Load an official or custom model
model = YOLO('yolov8n.pt')

# Perform tracking with the model
results = model.track(source="output7.mp4", show=True)
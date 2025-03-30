from ultralytics import YOLO

# Load the YOLOv8n model
model = YOLO('C:\\Users\\samne\\OneDrive\\ADVAY\\Food Waste\\TuneYolo\\yolo11n.pt')  # Update with the correct path to your model

# Export the model to TorchScript format
model.export(format='torchscript')  # This creates 'yolov8n.torchscript'

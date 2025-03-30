import torch
from ultralytics import YOLO

# Load your tuned model checkpoint using Ultralytics YOLO
model = YOLO("C:\\Users\\samne\\OneDrive\\ADVAY\\Food Waste\\TuneYolo\\YOLO_Model.pt")
model.eval()  # Ensure the model is in evaluation mode

# Create an example input tensor with the dimensions used during training (batch size 1, 3 channels, 128x128)
example_input = torch.randn(1, 3, 128, 128)

# For tracing, we need to trace the underlying model (access the underlying module with model.model)
traced_model = torch.jit.trace(model.model, example_input)

# Save the traced model to a TorchScript file
traced_model.save("model.torchscript.pt")

print("Model has been successfully converted to TorchScript and saved as model.torchscript.pt")

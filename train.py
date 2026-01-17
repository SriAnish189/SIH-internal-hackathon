from ultralytics import YOLO
import torch
import os

# -------------------------------
# 1. Check GPU availability
# -------------------------------
device = 0 if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# -------------------------------
# 2. Paths
# -------------------------------
# Make sure to replace this with your actual paths
data_yaml = "data.yaml"
pretrained_model = "yolov8m.pt"  # YOLOv8 medium pretrained weights
save_dir = "C:/Users/srian/OneDrive/Desktop/clg/train_results"

# Create save directory if it doesn't exist
os.makedirs(save_dir, exist_ok=True)

# -------------------------------
# 3. Initialize YOLOv8 model
# -------------------------------
model = YOLO(pretrained_model)

# -------------------------------
# 4. Train model
# -------------------------------
results = model.train(
    data=data_yaml,      # path to data.yaml
    epochs=20,           # number of training epochs
    batch=16,            # batch size
    imgsz=640,           # image size
    device=device,       # GPU or CPU
    name="train_results",# experiment name
    save=True,           # save best model
    save_period=10,      # save checkpoint every 10 epochs
    workers=4,           # number of data loader workers
    exist_ok=True        # overwrite existing experiment folder if exists
)

# -------------------------------
# 5. Print final metrics
# -------------------------------
print("Training complete!")
print("Results:", results)

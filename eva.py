from ultralytics import YOLO

# Paths to your weights
best_model_path = "C:\Users\srian\OneDrive\Desktop\clg\SIH25\yolo11n.pt"
last_model_path = "C:\Users\srian\OneDrive\Desktop\clg\SIH25\yolov8m.pt"
Load models
best_model = YOLO(best_model_path)
last_model = YOLO(last_model_path)

# Evaluate on validation dataset
# Make sure your data.yaml has 'val:' pointing to validation images
print("Evaluating best.pt ...")
best_metrics = best_model.val(data="/path/to/data.yaml")
print("\nEvaluating last.pt ...")
last_metrics = last_model.val(data="/path/to/data.yaml")

# Compare mAP50
best_map = best_metrics.box.map[0]   # mAP50
last_map = last_metrics.box.map[0]   # mAP50

print(f"\nbest.pt mAP50: {best_map}")
print(f"last.pt mAP50: {last_map}")

if best_map >= last_map:
    print("✅ Use best.pt for inference")
else:
    print("✅ Use last.pt for inference")

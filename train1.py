def main():
    # all your training code here
    print("Training starts now!")
    # example: call YOLO training
    from ultralytics import YOLO

    model = YOLO("yolov8m.pt")  # pretrained model
    model.train(
        data="C:/Users/srian/OneDrive/Desktop/clg/SIH25",
        epochs=20,
        batch=16,
        imgsz=640,
        device=0,
        name="train_results"
    )

if __name__ == "__main__":
    main()

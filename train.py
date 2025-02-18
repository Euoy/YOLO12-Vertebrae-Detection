from ultralytics import YOLO

if __name__ == "__main__":

    model = YOLO("yolo11s.pt")
    model.train(
        data="train_config.yaml",
        epochs=500,
        batch=16,
        imgsz=640
        )
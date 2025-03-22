from ultralytics import YOLO
from pathlib import Path

if __name__ == "__main__":

    project = f"{Path().absolute()}\\runs"
    model = YOLO("yolo11s.pt", project=project)
    model.train(
        data="train_config.yaml",
        epochs=500,
        batch=16,
        imgsz=640,
        project=project
        )
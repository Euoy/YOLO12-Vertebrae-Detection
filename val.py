from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("E:\\YOLO11-Vertebrae-Detection\\runs\\detect\\train5\\weights\\best.pt")

    results = model.val(data="val.yaml", save=True)
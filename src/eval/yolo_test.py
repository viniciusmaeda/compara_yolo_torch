from ultralytics import YOLO

model = YOLO("yolo26n.pt")

model.train(
    data="coco128.yaml",
    epochs=10,
    imgsz=640
)
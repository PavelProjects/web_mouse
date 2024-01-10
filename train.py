from ultralytics import YOLO

model = YOLO("yolov8n.yaml")

if __name__ == '__main__':
    model.train(data="data.yaml", epochs=150)

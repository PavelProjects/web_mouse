from ultralytics import YOLO

model = YOLO("/home/pobopo/labi/2kurs/ml/web_mouse/yolov8/exp4/weights/best.pt")
results = model.predict(source="0", show=True)
print(results)

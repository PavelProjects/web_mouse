import torch
import cv2 as cv

MODEL_PATH = '/home/pobopo/labi/2kurs/ml/web_mouse/weights/best1.pt'

model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_PATH)

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Failed to open video capture")
    exit(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    detection = model(frame)

    cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
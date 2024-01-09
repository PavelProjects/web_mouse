from ultralytics import YOLO
import cv2 as cv

def main():
  model = YOLO("/home/pobopo/labi/2kurs/ml/web_mouse/yolov8/exp1/weights/best.pt")

  capture = cv.VideoCapture(0)
  while True:
    try:
      ret, frame = capture.read()
      if (not ret):
        print("Stream end")
        return
      results = model.predict(source=frame)
      res_img = results[0].orig_img
      
      boxes = results[0].boxes
      if (len(boxes.cls) == 1):
        name = results[0].names[boxes.cls.item()]
        print(boxes.xyxy[0])
        xyxy = boxes.xyxy[0].tolist()
        print(f"Found {name}[{boxes.conf}]")
        cv.rectangle(res_img, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 1)

      cv.imshow("Prediction", res_img)
      key = cv.waitKey(1) & 0xFF
      if key == ord("q"):
          break

    except KeyboardInterrupt:
      print("Leaving")
      break

if __name__ == "__main__":
  main()
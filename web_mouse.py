from ultralytics import YOLO
import pyautogui as pg
import cv2 as cv

def mid(p1, p2):
  return int((p2 - p1) // 2 + p1)

def main(show=False):
  model = YOLO("/home/pobopo/labi/2kurs/ml/web_mouse/yolov8/exp4/weights/best.pt")

  cntr = (240, 320)
  prevCntr = cntr

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
        box = boxes.xyxy[0].tolist()

        cntr = (mid(box[0], box[2]), mid(box[1], box[3]))

        if (show):
          print(f"Found {name}[{boxes.conf}] box={box}, cntr={cntr}")
          cv.rectangle(res_img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
          cv.circle(res_img, cntr, 5, (0, 0, 255), -1)

      pg.move(prevCntr[0] - cntr[0], cntr[1] - prevCntr[1])
      prevCntr = cntr

      if (show):
        cv.imshow("Prediction", res_img)
        key = cv.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    except KeyboardInterrupt:
      print("Leaving")
      break

if __name__ == "__main__":
  main(True)
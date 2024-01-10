from ultralytics import YOLO
import pyautogui as pg
import cv2 as cv

# Average of hand center pos
CNTR_AVERAGE_OF = 5
CLASS_AVERAGE_OF = 5
SAVE_DIR = "/home/pobopo/Pictures/misses/"
INGORE_PRESS = True

# without this shit script slow af
pg.PAUSE = 0
# this thing disables usless error
pg.FAILSAFE = False

PAPER_CLASS = 0
ROCK_CLASS = 1
SCISSORS_CLASS = 2

FRAME_SIZE = (640, 480)
KOEF = 2 * (pg.resolution()[1] // FRAME_SIZE[1])

def main(show=False):
  model = YOLO("/home/pobopo/labi/2kurs/ml/web_mouse/yolov8/exp5(n+)/weights/best.pt")

  cntr = None
  prevCntr = cntr
  cntrHistory = None

  clsHistory = None
  detectedCls = -1

  buttonPressed = None
  missSaved = False
  i = 0

  capture = cv.VideoCapture(0)
  capture.set(cv.CAP_PROP_FRAME_WIDTH, FRAME_SIZE[0])
  capture.set(cv.CAP_PROP_FRAME_HEIGHT, FRAME_SIZE[1])

  try:
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
          if (clsHistory == None):
            clsHistory = [boxes.cls.item()] * (CLASS_AVERAGE_OF - 1)
          elif (len(clsHistory) >= CLASS_AVERAGE_OF):
            clsHistory.pop(0)
          clsHistory.append(boxes.cls.item())
          detectedCls = sum(clsHistory) // CLASS_AVERAGE_OF

          name = results[0].names[detectedCls]
          box = boxes.xyxy[0].tolist()

          cntr = (mid(box[0], box[2]), mid(box[1], box[3]))
          if (cntrHistory == None):
            cntrHistory = [cntr] * (CNTR_AVERAGE_OF - 1)
          elif (len(cntrHistory) >= CNTR_AVERAGE_OF):
            cntrHistory.pop(0)
          cntrHistory.append(cntr)

          sumXY = [0, 0]
          for x, y in cntrHistory:
            sumXY[0] += + x
            sumXY[1] += + y
          cntr = (sumXY[0] // CNTR_AVERAGE_OF, sumXY[1] // CNTR_AVERAGE_OF)

          if (prevCntr == None):
            prevCntr = cntr

          if (show):
            print(f"Found {name}[{boxes.conf}] box={box}, cntr={cntr}")
            cv.rectangle(res_img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
            cv.circle(res_img, cntr, 5, (0, 0, 255), -1)

          missSaved = False
        elif (SAVE_DIR != None and not missSaved):
          cv.imwrite(f"{SAVE_DIR}miss{i}.jpg", res_img)
          i += 1
          missSaved = True


        if (detectedCls == -1):
          pass
        else:
          pg.move(KOEF * (prevCntr[0] - cntr[0]), KOEF * (cntr[1] - prevCntr[1]))
          # bad
          if (not INGORE_PRESS and detectedCls in [SCISSORS_CLASS, ROCK_CLASS]):
            if (detectedCls == ROCK_CLASS):
              buttonPressed = "left"
            elif (detectedCls == SCISSORS_CLASS):
              buttonPressed = "right"
            pg.mouseDown(button=buttonPressed)
          elif (buttonPressed != None):
            pg.mouseUp(button=buttonPressed)
            buttonPressed = None
            
        prevCntr = cntr

        if (show):
          cv.imshow("Cool mouse", res_img)
          key = cv.waitKey(1) & 0xFF
          if key == ord("q"):
              break
      except KeyboardInterrupt:
        break
  finally:
    print("Leaving")
    if (SAVE_DIR != None):
      print(f"Saved {i} misses")
    capture.release()
    pg.mouseUp()
    pg.mouseUp(button="right")

def mid(p1, p2):
  return int((p2 - p1) // 2 + p1)

if __name__ == "__main__":
  main(True)
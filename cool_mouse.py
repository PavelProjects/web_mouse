from ultralytics import YOLO
import pyautogui as pg
import cv2 as cv
from time import time

MODEL_CONF = 0.25
# Average of hand center pos
CNTR_AVERAGE_OF = 3
CLASS_AVERAGE_OF = 5
SAVE_DIR = "/home/pobopo/Pictures/misses/"

CLICK_DISABLED = True
SCROLL_DISABLED = True
CLICK_HOLD_TIME = 1
SCROLL_HOLD_TIME = 1

# without this shit script slow af
pg.PAUSE = 0
# this thing disables usless error
pg.FAILSAFE = False

# default classes
PAPER_CLASS = -1
POINTER_CLASS = -1
ROCK_CLASS = -1
SCISSORS_CLASS = -1

FRAME_SIZE = (640, 480)
KOEF = 2 * (pg.resolution()[1] // FRAME_SIZE[1])

def main(show=False):
  model = YOLO("/home/pobopo/labi/2kurs/ml/web_mouse/yolov8/exp5(n+)/weights/best.pt")

  for (key, name) in model.names.items():
    if (name == "Paper"):
      PAPER_CLASS = key
    elif (name == "Rock"):
      ROCK_CLASS = key
    elif (name == "Scissors"):
      SCISSORS_CLASS = key
    elif (name == "Pointer"):
      POINTER_CLASS = key

  cntr = None
  prevCntr = cntr
  cntrHistory = None

  clsHistory = None
  detectedCls = -1
  prevCls = -1

  rockHoldStart = None
  sciccorsHoldStart = None
  moveDisabled = True
  buttonPressed = False
  missSaved = False
  i = 0

  capture = cv.VideoCapture(0)
  capture.set(cv.CAP_PROP_FRAME_WIDTH, FRAME_SIZE[0])
  capture.set(cv.CAP_PROP_FRAME_HEIGHT, FRAME_SIZE[1])

  frameText = ""
  box = None

  try:
    while True:
      try:
        ret, frame = capture.read()
        if (not ret):
          print("Stream end")
          return

        results = model.predict(source=frame, conf=MODEL_CONF)
        res_img = results[0].orig_img
        
        boxes = results[0].boxes
        if (len(boxes.cls) == 1):
          if (clsHistory == None):
            clsHistory = [boxes.cls.item()] * (CLASS_AVERAGE_OF - 1)
          elif (len(clsHistory) >= CLASS_AVERAGE_OF):
            clsHistory.pop(0)
          clsHistory.append(boxes.cls.item())
          prevCls = detectedCls
          detectedCls = sum(clsHistory) // CLASS_AVERAGE_OF

          box = boxes.xyxy[0].tolist()
          # cntr runs away when u change hand size
          # cntr = (mid(box[0], box[2]), mid(box[1], box[3])) 
          cntr = (int(box[0]), int(box[3]))
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

          missSaved = False
        elif (SAVE_DIR != None and not missSaved):
          cv.imwrite(f"{SAVE_DIR}miss{i}.jpg", res_img)
          i += 1
          missSaved = True

        # BRUH
        if (detectedCls == -1 or prevCls == -1):
          frameText = "skip"
          pass
        elif (detectedCls == PAPER_CLASS):
          moveDisabled = False
        elif (detectedCls == SCISSORS_CLASS):
          moveDisabled = True
          if (not SCROLL_DISABLED):
            if (sciccorsHoldStart == None):
              sciccorsHoldStart = time()
            elif (time() - sciccorsHoldStart > SCROLL_HOLD_TIME):
              pg.scroll((cntr[1] - prevCntr[1]) // KOEF)
              frameText = "scroll"
        elif (prevCls == SCISSORS_CLASS and prevCls != detectedCls):
          sciccorsHoldStart = None
        elif (not CLICK_DISABLED):
          if (prevCls == ROCK_CLASS and prevCls != detectedCls):
            if (rockHoldStart != None and time() - rockHoldStart <= CLICK_HOLD_TIME):
              pg.click()
              moveDisabled = False
              frameText = "click"
            pg.mouseUp()
            buttonPressed = False
            rockHoldStart = None

          if (detectedCls == ROCK_CLASS):
            if (rockHoldStart == None):
              rockHoldStart = time()
              moveDisabled = True
            elif (time() - rockHoldStart > CLICK_HOLD_TIME):
              pg.mouseDown()
              buttonPressed = True
              moveDisabled = False

          if (buttonPressed):
            frameText = "mouseDown"
          if (moveDisabled):
            frameText = "hold"

        if (not moveDisabled):
          pg.move(KOEF * (prevCntr[0] - cntr[0]), KOEF * (cntr[1] - prevCntr[1]))
            
        prevCntr = cntr

        if (show):
          if (box != None):
            coords = ((int(box[0]), int(box[1])), (int(box[2]), int(box[3])))
            cv.rectangle(res_img, coords[0], coords[1], (0, 255, 0), 2)
            cv.circle(res_img, cntr, 5, (0, 0, 255), -1)
            text = f"{results[0].names[detectedCls]}({boxes.conf})"
            cv.putText(
              res_img, 
              text,
              (coords[0][0], coords[0][1] - 10),
              cv.FONT_ITALIC, 0.5, (0, 255, 0), 2, 1
            )
            if (len(frameText) > 0):
              cv.putText(
                res_img,
                frameText,
                (cntr[0] - len(frameText), cntr[1] + 15),
                cv.FONT_ITALIC, 0.5, (255, 0, 0), 2, 1
              )
              frameText = ""

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

def mid(p1, p2):
  return int((p2 - p1) // 2 + p1)

if __name__ == "__main__":
  main(True)
import cv2
import numpy as np
from math import dist
from pathlib import Path
import json
import random

colors = ["red", "yellow", "green"]
radnom_colors = random.shuffle(colors)

save_path = Path(__file__).parent
config_path = save_path / "config.json"

cv2.namedWindow("Image", cv2.WINDOW_GUI_NORMAL)
cv2.namedWindow("Mask", cv2.WINDOW_GUI_NORMAL)

position = [0, 0]
clicked = False

def on_click(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Clicked 1 at {x}, {y}")     
    elif event == cv2.EVENT_RBUTTONDOWN:
        print(f"Clicked 2 at {x}, {y}")
    elif event == cv2.EVENT_RBUTTONUP:
        print(f"Clicked 3 at {x}, {y}")
    
        global position
        global clicked
        position = [x, y]
        clicked = True

cv2.setMouseCallback("Image", on_click)

cam = cv2.VideoCapture(0)

lower = None
upper = None
lower1 = None
upper1 = None
lower2 = None
upper2 = None

if config_path.exists():
    with config_path.open("r") as f:
        js = json.load(f)
        if "lower" in js:
            lower = np.array(js["lower"], dtype="u1")
            upper = np.array(js["upper"], dtype="u1")
        elif "lower1" in js:
            lower1 = np.array(js["lower1"], dtype="u1")
            upper1 = np.array(js["upper1"], dtype="u1")
        elif "lower2" in js:
            lower2 = np.array(js["lower2"], dtype="u1")
            upper2 = np.array(js["upper2"], dtype="u1")

while cam.isOpened():
    ret, frame = cam.read()
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    if clicked:
        clicked = False
        color = hsv[position[1], position[0]]

        lower = np.clip(color * 0.9, 0, 255).astype("u1")
        upper = np.clip(color * 1.1, 0, 255).astype("u1")
        lower1 = np.clip(color * 0.9, 0, 255).astype("u1")
        upper1 = np.clip(color * 1.1, 0, 255).astype("u1")
        lower2 = np.clip(color * 0.9, 0, 255).astype("u1")
        upper2 = np.clip(color * 1.1, 0, 255).astype("u1")
        # upper[1] = 255
        # upper[2] = 255
    if lower is not None and lower1 is not None and lower2 is not None:
        inr = cv2.inRange(hsv, lower, upper)
        inr1 = cv2.inRange(hsv, lower1, upper1)
        inr2 = cv2.inRange(hsv, lower2, upper2)

        mask = cv2.morphologyEx(inr, cv2.MORPH_CLOSE, np.ones((5, 5), dtype="u1"))
        mask1 = cv2.morphologyEx(inr1, cv2.MORPH_CLOSE, np.ones((5, 5), dtype="u1"))
        mask2 = cv2.morphologyEx(inr2, cv2.MORPH_CLOSE, np.ones((5, 5), dtype="u1"))

        cv2.imshow("Mask", mask)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours1, _ = cv2.findContours(mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours2, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        positions = []
        if len(contours) > 0:
            contour = max(contours, key=cv2.contourArea)
            (x, y), radius = cv2.minEnclosingCircle(contour)
            if radius > 10:
                x = int(x)
                y = int(y)
                radius = int(radius)
                cv2.circle(frame, (x, y), radius, (0, 255, 255), 4)
                cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
                positions.append((x, y))
                if len(positions) > 20:
                    positions.pop(0)
                for i, position in enumerate(positions[:-1]):
                    cv2.circle(frame, position, i *  2, (0, 0, 100 + 155 / len(positions) * i), -1)

        if len(contours1) > 0:
            contour = max(contours1, key=cv2.contourArea)
            (x, y), radius = cv2.minEnclosingCircle(contour)
            if radius > 10:
                x = int(x)
                y = int(y)
                radius = int(radius)
                cv2.circle(frame, (x, y), radius, (0, 255, 255), 4)
                cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
                positions.append((x, y))
                if len(positions) > 20:
                    positions.pop(0)
                for i, position1 in enumerate(positions[:-1]):
                    cv2.circle(frame, position1, i *  2, (0, 0, 100 + 155 / len(positions) * i), -1)

        if len(contours2) > 0:
            contour = max(contours2, key=cv2.contourArea)
            (x, y), radius = cv2.minEnclosingCircle(contour)
            if radius > 10:
                x = int(x)
                y = int(y)
                radius = int(radius)
                cv2.circle(frame, (x, y), radius, (0, 255, 255), 4)
                cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
                positions.append((x, y))
                if len(positions) > 20:
                    positions.pop(0)
                for i, position2 in enumerate(positions[:-1]):
                    cv2.circle(frame, position2, i *  2, (0, 0, 100 + 155 / len(positions) * i), -1)
            

    cv2.imshow("Image", frame)

cam.release()
cv2.destroyAllWindows()

with config_path.open("w") as f:
    json.dump(
        {"lower": None if lower is None else lower.tolist(),
         "upper": None if upper is None else upper.tolist(),
         "lower1": None if lower1 is None else lower1.tolist(),
         "upper1": None if upper1 is None else upper1.tolist(),
         "lower2": None if lower2 is None else lower1.tolist(),
         "upper2": None if upper2 is None else upper1.tolist()
         }, 
        f
    )
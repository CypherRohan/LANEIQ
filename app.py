from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import os
from ultralytics import YOLO
from sort import Sort

app = FastAPI()

# Load YOLO model and SORT tracker once
model = YOLO('last.pt')
tracker = Sort()

# Constants
STOP_LINE_Y = 400

# Helmet and bike ID sets
helmet_ids = set()
bike_ids = set()

# Violation folder
if not os.path.exists('violations'):
    os.makedirs('violations')


def process_frame_logic(frame):
    # --- Processing Frame Logic (Same as your main code) ---

    zebra_box = (100, STOP_LINE_Y - 50, frame.shape[1]-100, STOP_LINE_Y + 10)

    # Detect traffic light
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([179, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = mask1 | mask2
    red_pixels = cv2.countNonZero(red_mask)
    traffic_light = "red" if red_pixels > 500 else "green"

    detections = model.predict(frame, stream=True)
    boxes = []
    confidences = []
    class_ids = []

    for result in detections:
        for box, cls, conf in zip(result.boxes.xyxy.cpu().numpy(),
                                  result.boxes.cls.cpu().numpy(),
                                  result.boxes.conf.cpu().numpy()):
            x1, y1, x2, y2 = box
            boxes.append([x1, y1, x2, y2])
            confidences.append(conf)
            class_ids.append(int(cls))

    dets_for_tracker = []
    for box, conf in zip(boxes, confidences):
        x1, y1, x2, y2 = box
        dets_for_tracker.append([x1, y1, x2, y2, conf])

    dets_for_tracker = np.array(dets_for_tracker)
    tracks = tracker.update(dets_for_tracker)

    result_data = []

    for track in tracks:
        x1, y1, x2, y2, track_id = track
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        for b, cid in zip(boxes, class_ids):
            bx1, by1, bx2, by2 = b
            if not (x1 > bx2 or x2 < bx1 or y1 > by2 or y2 < by1):
                class_found = cid
                break
        else:
            continue

        obj = {
            "track_id": int(track_id),
            "class_id": int(class_found),
            "bbox": [float(x1), float(y1), float(x2), float(y2)],
            "traffic_light": traffic_light
        }
        result_data.append(obj)

    return result_data


@app.post("/process-frame")
async def process_frame(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        detections = process_frame_logic(frame)

        return JSONResponse(content=detections)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/")
def home():
    return {"message": "Backend is running!"}

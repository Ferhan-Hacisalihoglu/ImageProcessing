# import libraries
import cv2
import numpy as np
from ultralytics import YOLO

# define healper functions
def get_line_side(x,y,lineStart, lineEnd):
    return np.sign((lineEnd[0] - lineStart[0]) * (y - lineStart[1]) - (lineEnd[1] - lineStart[1]) * (x - lineStart[0]))

# define model
model = YOLO("models/yolov8s.pt")

# video capture
cap = cv2.VideoCapture("data/HumanActivityRecognitionDatasetOpenPose/2.mp4")

success, frame = cap.read()
if not success:
    exit("Failed to read video")

frame_height, frame_width = frame.shape[:2]

resize_scale = 0.6
resized_height = int(frame_height * resize_scale)
resized_width = int(frame_width * resize_scale)

# define line
line_x = int(resized_width * 0.5)
line_start = (line_x, resized_height-1)
line_end = (line_x, 1)

# counters
inputPeople = 0
outputPeople = 0
counterd_ids = set()
person_last_x = {}

# count people with yolo
while True:

    success, frame = cap.read()
    if not success:
        break

    frame = cv2.resize(frame, (0, 0), fx=resize_scale, fy=resize_scale)

    results = model.track(frame, persist=True, stream=False, conf = 0.3, iou = 0.3, tracker = "bytetrack.yaml")

    if results[0].boxes.id is not None:
        ids = results[0].boxes.id.int().tolist()
        classes = results[0].boxes.cls.int().tolist()
        xyxy = results[0].boxes.xyxy

        for i, box in enumerate(xyxy):
            cls_id = classes[i]
            track_id = ids[i]
            class_name = model.names[cls_id]

            if class_name != "person":
                continue

            x1, y1, x2, y2 = map(int, box)
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            previous_x = person_last_x.get(track_id, None)
            person_last_x[track_id] = cx

            if previous_x is not None:
                if previous_x > line_x >= cx:
                    if track_id not in counterd_ids:
                        outputPeople += 1
                        counterd_ids.add(track_id)
                elif previous_x < line_x <= cx:
                    if track_id not in counterd_ids:
                        inputPeople += 1
                        counterd_ids.add(track_id)

            # draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

    cv2.line(frame, line_start, line_end, (0, 0, 255), 2)
    cv2.putText(frame, f"Input: {inputPeople}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Output: {outputPeople}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("People Counter", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



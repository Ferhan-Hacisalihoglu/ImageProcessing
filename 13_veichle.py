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
cap = cv2.VideoCapture("data/CarDetactionVideos/IMG_5268.MOV")

success, frame = cap.read()
if not success:
    exit("Failed to read video")

frame_height, frame_width = frame.shape[:2]

resize_scale = 0.6
resized_height = int(frame_height * resize_scale)
resized_width = int(frame_width * resize_scale)

# define line
line_start = (int(resized_width * 0.5), resized_height - 1)
line_end = (resized_width - 1, int(resized_height * 0.2))

# object types
counts = {"car": 0,"truck": 0, "bus": 0,"motorcycle": 0,"bicycle": 0}
counted_ids = set()
object_last_side = {}

# count cars with yolo
while True:

    success, frame = cap.read()
    if not success:
        exit("Failed to read video")

    frame = cv2.resize(frame, (0, 0), fx=resize_scale, fy=resize_scale)

    results = model.track(frame, persist=True, stream=False, conf = 0.5, iou = 0.5, tracker = "bytetrack.yaml")

    if results[0].boxes.id is not None:
        ids = results[0].boxes.id.int().tolist()
        classes = results[0].boxes.cls.int().tolist()
        xyxy = results[0].boxes.xyxy

        for i, box in enumerate(xyxy):
            cls_id = classes[i]
            track_id = ids[i]
            class_name = model.names[cls_id]
            if class_name not in counts:
                continue
            x1, y1, x2, y2 = map(int, box)
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2) 

            current_side = get_line_side(cx, cy, line_start, line_end)
            previous_side = object_last_side.get(track_id, None)
            object_last_side[track_id] = current_side

            if previous_side is not None and current_side != previous_side:
                if track_id not in counted_ids:
                    counts[class_name] += 1
                    counted_ids.add(track_id)

            # draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{class_name} ID:{track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2) 
            cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

    cv2.line(frame, line_start, line_end, (0, 0, 255), 2)

    # show counters
    y_offset = 30
    for cls,count in counts.items():
        text = f"{cls}: {count}"
        cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2) 
        y_offset += 30

    cv2.imshow("Car Counter", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
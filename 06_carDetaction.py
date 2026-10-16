from ultralytics import YOLO
import cv2

model = YOLO("models/yolov8s.pt")

videos_path = "data/CarDetactionVideos/IMG_5270.MOV"
cap = cv2.VideoCapture(videos_path)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter("results/CarDetactionVideos.avi", cv2.VideoWriter_fourcc(*"XVID"), fps, (width, height))

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    results = model.track(
        frame, # input image
        persist=True, # protects the object ID from changing across frames
        conf=0.3, # minimum confidence threshold for object detection
        iou=0.5, # minimum intersection over union threshold for object tracking
        tracker = "bytetrack.yaml", # path to the tracker configuration file
    ) 

    annotated_frame = results[0].plot()

    cv2.imshow("YoloV8 Tracking", annotated_frame)
    out.write(annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
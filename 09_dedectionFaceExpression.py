# import libraries
import cv2

import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

# define heyperparameters

MODEL_PATH = "models/face_landmarker.task"
CAMERA_INDEX = 0

# define helper functions

def classify_expression(blendshapes):
    scores = {category.category_name: category.score for category in blendshapes}
    smile = (scores.get("mouthSmileLeft", 0.0) + scores.get("mouthSmileRight", 0.0)) / 2
    frown = (scores.get("mouthFrownLeft", 0.0) + scores.get("mouthFrownRight", 0.0)) / 2
    surprise = scores.get("jawOpen", 0.0)
    brow_raise = (
        scores.get("browInnerUp", 0.0)
        + scores.get("browOuterUpLeft", 0.0)
        + scores.get("browOuterUpRight", 0.0)
    ) / 3

    candidates = {
        "Happy": smile,
        "Sad": frown,
        "Surprised": min(1.0, (surprise + brow_raise) / 2),
    }
    expression, confidence = max(candidates.items(), key=lambda item: item[1])
    if confidence < 0.35:
        return "Neutral", 1.0 - max(candidates.values())
    return expression, confidence


def draw_face(frame, landmarks):
    height, width = frame.shape[:2]
    points = [
        (int(landmark.x * width), int(landmark.y * height))
        for landmark in landmarks
    ]

    for connection in vision.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS:
        cv2.line(
            frame,
            points[connection.start],
            points[connection.end],
            (0, 255, 0),
            1,
        )

# process video and predict face expression

cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    raise RuntimeError(f"Could not open camera {CAMERA_INDEX}")

options = vision.FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    output_face_blendshapes=True,
)

frame_index = 0
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

with vision.FaceLandmarker.create_from_options(options) as landmarker:
    while True:
        success, frame = cap.read()
        if not success:
            break

        timestamp_ms = int(frame_index * 1000 / fps)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        if result.face_landmarks:
            draw_face(frame, result.face_landmarks[0])
            expression, confidence = classify_expression(result.face_blendshapes[0])
            text = f"Expression: {expression} ({confidence:.0%})"
        else:
            text = "Expression: No face detected"

        cv2.putText(
            frame,
            text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
        cv2.imshow("Face expression detection", frame)

        frame_index += 1
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()

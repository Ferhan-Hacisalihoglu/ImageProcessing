# import libraries
import math
import cv2

import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

# define hyperparameters

MODEL_PATH = "models/pose_landmarker_lite.task"
VIDEO_PATH = "data/HumanActivityRecognitionDatasetOpenPose/squat_test1.avi"
SQUAT_DOWN_ANGLE = 90
SQUAT_UP_ANGLE = 160

# define helper functions

def calculate_angle(a, b, c):
    """Calculate the angle at point b for three 2D points."""
    radians = math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(
        a[1] - b[1], a[0] - b[0]
    )
    angle = abs(math.degrees(radians))
    return 360 - angle if angle > 180 else angle


def draw_pose(frame, landmarks):
    height, width = frame.shape[:2]
    points = [
        (int(landmark.x * width), int(landmark.y * height))
        for landmark in landmarks
    ]

    for connection in vision.PoseLandmarksConnections.POSE_LANDMARKS:
        start = points[connection.start]
        end = points[connection.end]
        cv2.line(frame, start, end, (0, 255, 0), 2)

    for point in points:
        cv2.circle(frame, point, 4, (0, 0, 255), -1)

# process video and predict pose

options = vision.PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
    running_mode=vision.RunningMode.VIDEO,
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

cap = cv2.VideoCapture(str(VIDEO_PATH))

fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
frame_index = 0
squat_count = 0
squat_down = False

with vision.PoseLandmarker.create_from_options(options) as landmarker:
    while True:
        success, frame = cap.read()
        if not success:
            break

        timestamp_ms = int(frame_index * 1000 / fps)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        if result.pose_landmarks:
            landmarks = result.pose_landmarks[0]
            draw_pose(frame, landmarks)

            left_knee = landmarks[vision.PoseLandmark.LEFT_KNEE]
            left_hip = landmarks[vision.PoseLandmark.LEFT_HIP]
            left_ankle = landmarks[vision.PoseLandmark.LEFT_ANKLE]
            knee_angle = calculate_angle(
                (left_hip.x, left_hip.y),
                (left_knee.x, left_knee.y),
                (left_ankle.x, left_ankle.y),
            )

            if knee_angle < SQUAT_DOWN_ANGLE:
                squat_down = True
            elif squat_down and knee_angle > SQUAT_UP_ANGLE:
                squat_count += 1
                squat_down = False

        cv2.putText(
            frame,
            f"Squats: {squat_count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
        )

        cv2.imshow("Pose prediction", frame)
        frame_index += 1

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()

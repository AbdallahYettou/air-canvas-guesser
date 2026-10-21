"""MediaPipe's pretrained hand landmarks; no gesture model training needed."""

from pathlib import Path
import cv2
import numpy as np
from core.gestures import Gesture, classify_landmarks

CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
               (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
               (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)]


class HandTracker:
    def __init__(self, model_path):
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        if not Path(model_path).is_file():
            raise FileNotFoundError("Hand model missing. Run: python scripts/download_hand_model.py")
        self.mp = mp
        self.last_timestamp = -1
        self.tracker = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO, num_hands=1,
            min_hand_detection_confidence=0.6, min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.6,
        ))

    def detect(self, frame, now):
        timestamp = max(self.last_timestamp + 1, int(now * 1000))
        self.last_timestamp = timestamp
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.tracker.detect_for_video(self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb), timestamp)
        if not result.hand_landmarks:
            return Gesture.IDLE, None, None
        height, width = frame.shape[:2]
        points = np.array([(p.x * width, p.y * height) for p in result.hand_landmarks[0]])
        pose = points
        if result.hand_world_landmarks:
            pose = np.array([(p.x, p.y, p.z) for p in result.hand_world_landmarks[0]])
        return classify_landmarks(pose), tuple(points[8]), points

    def close(self):
        self.tracker.close()


def draw_landmarks(frame, points):
    if points is None:
        return
    points = np.rint(points).astype(int)
    for start, end in CONNECTIONS:
        cv2.line(frame, tuple(points[start]), tuple(points[end]), (175, 155, 95), 1, cv2.LINE_AA)
    for point in points:
        cv2.circle(frame, tuple(point), 3, (155, 235, 200), -1, cv2.LINE_AA)

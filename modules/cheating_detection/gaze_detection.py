import collections

import cv2
import mediapipe as mp
import numpy as np


class RobustGazeTracker:
    def __init__(self, yaw_threshold=18, pitch_threshold=15, smoothing_frames=5):
        self.yaw_threshold = yaw_threshold
        self.pitch_threshold = pitch_threshold
        self.smoothing_frames = smoothing_frames

        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3
        )

        self.yaw_buffer = collections.deque(maxlen=smoothing_frames)
        self.pitch_buffer = collections.deque(maxlen=smoothing_frames)

        self.landmark_ids = {
            "nose": 1,
            "chin": 152,
            "left_eye_outer": 33,
            "right_eye_outer": 263,
            "left_mouth": 61,
            "right_mouth": 291,
            "left_cheek": 234,
            "right_cheek": 454,
            "forehead": 10
        }

    def _point(self, landmarks, idx, w, h):
        return np.array([landmarks[idx].x * w, landmarks[idx].y * h], dtype=np.float64)

    def _estimate_yaw_pitch(self, landmarks, w, h):
        nose = self._point(landmarks, self.landmark_ids["nose"], w, h)
        chin = self._point(landmarks, self.landmark_ids["chin"], w, h)
        left_eye = self._point(landmarks, self.landmark_ids["left_eye_outer"], w, h)
        right_eye = self._point(landmarks, self.landmark_ids["right_eye_outer"], w, h)
        left_mouth = self._point(landmarks, self.landmark_ids["left_mouth"], w, h)
        right_mouth = self._point(landmarks, self.landmark_ids["right_mouth"], w, h)
        left_cheek = self._point(landmarks, self.landmark_ids["left_cheek"], w, h)
        right_cheek = self._point(landmarks, self.landmark_ids["right_cheek"], w, h)
        forehead = self._point(landmarks, self.landmark_ids["forehead"], w, h)

        left_width = max(nose[0] - left_cheek[0], 1.0)
        right_width = max(right_cheek[0] - nose[0], 1.0)
        yaw_ratio = (left_width - right_width) / max(left_width + right_width, 1.0)
        yaw = float(np.clip(yaw_ratio * 120.0, -90.0, 90.0))

        eye_mid = (left_eye + right_eye) / 2.0
        mouth_mid = (left_mouth + right_mouth) / 2.0

        upper_face = max(nose[1] - forehead[1], 1.0)
        lower_face = max(chin[1] - nose[1], 1.0)
        vertical_ratio = (upper_face - lower_face) / max(upper_face + lower_face, 1.0)

        eye_to_mouth = max(mouth_mid[1] - eye_mid[1], 1.0)
        nose_offset = ((nose[1] - eye_mid[1]) / eye_to_mouth) - 0.52

        pitch = float(np.clip((vertical_ratio * 140.0) + (nose_offset * 90.0), -90.0, 90.0))
        return yaw, pitch

    def process_frame(self, frame):
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return {"success": False}

        landmarks = results.multi_face_landmarks[0].landmark
        yaw, pitch = self._estimate_yaw_pitch(landmarks, w, h)

        self.yaw_buffer.append(yaw)
        self.pitch_buffer.append(pitch)
        smooth_yaw = float(np.mean(self.yaw_buffer))
        smooth_pitch = float(np.mean(self.pitch_buffer))

        looking_away = abs(smooth_yaw) > self.yaw_threshold or abs(smooth_pitch) > self.pitch_threshold

        gaze_dir = "center"
        if abs(smooth_yaw) >= abs(smooth_pitch) and abs(smooth_yaw) > self.yaw_threshold:
            gaze_dir = "right" if smooth_yaw > 0 else "left"
        elif abs(smooth_pitch) > self.pitch_threshold:
            gaze_dir = "down" if smooth_pitch > 0 else "up"

        return {
            "success": True,
            "yaw": smooth_yaw,
            "pitch": smooth_pitch,
            "looking_away": looking_away,
            "gaze_direction": gaze_dir
        }

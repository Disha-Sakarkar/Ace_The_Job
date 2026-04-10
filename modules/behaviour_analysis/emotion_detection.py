import logging
from collections import Counter, deque
from enum import Enum
from math import hypot

import cv2
import mediapipe as mp

logger = logging.getLogger(__name__)


class EmotionState(Enum):
    CONFIDENT = "confident"
    NEUTRAL = "neutral"
    NERVOUS = "nervous"


class FacialEmotionAnalyzer:
    def __init__(self, model_type="emotion", confidence_threshold=0.6):
        self.confidence_threshold = confidence_threshold
        self.model_type = model_type
        self.state_history = deque(maxlen=5)
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.emotion_model = self._load_emotion_model()
        self.emotion_to_state = {
            "happy": EmotionState.CONFIDENT,
            "neutral": EmotionState.NEUTRAL,
            "sad": EmotionState.NERVOUS,
            "angry": EmotionState.NERVOUS,
            "surprise": EmotionState.CONFIDENT,
            "fear": EmotionState.NERVOUS,
            "disgust": EmotionState.NERVOUS,
            "confident": EmotionState.CONFIDENT,
            "nervous": EmotionState.NERVOUS,
        }

    def _load_emotion_model(self):
        """Use FER/DeepFace when they load correctly, otherwise use landmark heuristics."""
        try:
            from fer.fer import FER

            logger.info("Emotion model: FER")
            return FER(use_tflite=True)
        except Exception as exc:
            logger.warning("FER unavailable, falling back to landmark analyzer: %s", exc)

        try:
            from deepface import DeepFace

            logger.info("Emotion model: DeepFace")
            return DeepFace
        except Exception as exc:
            logger.warning("DeepFace unavailable, using landmark analyzer: %s", exc)

        logger.info("Emotion model: MediaPipe landmark heuristic")
        return self._landmark_emotion_detector

    @staticmethod
    def _distance(a, b):
        return hypot(a.x - b.x, a.y - b.y)

    def _extract_features(self, landmarks):
        face_width = max(self._distance(landmarks[234], landmarks[454]), 1e-6)
        face_height = max(self._distance(landmarks[10], landmarks[152]), 1e-6)

        mouth_width = self._distance(landmarks[61], landmarks[291]) / face_width
        mouth_open = self._distance(landmarks[13], landmarks[14]) / face_height
        left_eye_open = self._distance(landmarks[159], landmarks[145]) / face_height
        right_eye_open = self._distance(landmarks[386], landmarks[374]) / face_height
        eye_open = (left_eye_open + right_eye_open) / 2.0
        brow_left = self._distance(landmarks[105], landmarks[159]) / face_height
        brow_right = self._distance(landmarks[334], landmarks[386]) / face_height
        brow_raise = (brow_left + brow_right) / 2.0
        lip_stretch = self._distance(landmarks[78], landmarks[308]) / face_width

        return {
            "mouth_width": mouth_width,
            "mouth_open": mouth_open,
            "eye_open": eye_open,
            "brow_raise": brow_raise,
            "lip_stretch": lip_stretch,
        }

    def _score_landmark_state(self, features):
        confident_score = 0.0
        nervous_score = 0.0

        if features["mouth_width"] > 0.38:
            confident_score += min(1.0, (features["mouth_width"] - 0.38) * 6.0)
        if features["lip_stretch"] > 0.44:
            confident_score += min(0.8, (features["lip_stretch"] - 0.44) * 6.0)
        if 0.012 < features["mouth_open"] < 0.075:
            confident_score += 0.2

        if features["eye_open"] > 0.06:
            nervous_score += min(1.0, (features["eye_open"] - 0.06) * 12.0)
        if features["brow_raise"] > 0.09:
            nervous_score += min(0.8, (features["brow_raise"] - 0.09) * 12.0)
        if features["mouth_open"] > 0.08:
            nervous_score += min(0.8, (features["mouth_open"] - 0.08) * 10.0)
        if features["mouth_width"] < 0.32 and features["eye_open"] > 0.055:
            nervous_score += 0.25

        if confident_score >= nervous_score + 0.2 and confident_score >= 0.35:
            return EmotionState.CONFIDENT.value, min(0.99, 0.5 + confident_score / 2.0)
        if nervous_score >= confident_score + 0.15 and nervous_score >= 0.35:
            return EmotionState.NERVOUS.value, min(0.99, 0.5 + nervous_score / 2.0)

        neutrality_margin = max(0.0, 1.0 - max(confident_score, nervous_score))
        return EmotionState.NEUTRAL.value, min(0.95, 0.55 + neutrality_margin * 0.3)

    def _stabilize_state(self, state, confidence):
        self.state_history.append(state)
        majority_state, count = Counter(self.state_history).most_common(1)[0]
        support = count / len(self.state_history)

        if majority_state != state and support >= 0.6:
            return majority_state, max(0.5, confidence * support)

        return state, confidence

    def _landmark_emotion_detector(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            return {"success": False}

        landmarks = results.multi_face_landmarks[0].landmark
        features = self._extract_features(landmarks)
        state, confidence = self._score_landmark_state(features)
        state, confidence = self._stabilize_state(state, confidence)

        return {
            "success": True,
            "dominant_emotion": state,
            "confidence": round(float(confidence), 3),
            "features": features,
        }

    def process_frame(self, frame):
        """
        Returns dict with keys: 'success', 'dominant_emotion' (string), 'confidence' (float)
        """
        if frame is None or frame.size == 0:
            return {"success": False}

        try:
            if hasattr(self.emotion_model, "detect_emotions"):  # FER
                result = self.emotion_model.detect_emotions(frame)
                if not result:
                    return {"success": False}

                emotions = result[0]["emotions"]
                dominant = max(emotions, key=emotions.get)
                confidence = float(emotions[dominant])
                state = self.emotion_to_state.get(dominant.lower(), EmotionState.NEUTRAL).value
                state, confidence = self._stabilize_state(state, confidence)
                return {
                    "success": True,
                    "dominant_emotion": state,
                    "confidence": round(float(confidence), 3),
                }

            if hasattr(self.emotion_model, "analyze"):  # DeepFace
                result = self.emotion_model.analyze(
                    frame,
                    actions=["emotion"],
                    enforce_detection=False,
                )
                if isinstance(result, list):
                    result = result[0] if result else None
                if not result:
                    return {"success": False}

                dominant = result["dominant_emotion"]
                confidence = float(result["emotion"][dominant]) / 100.0
                state = self.emotion_to_state.get(dominant.lower(), EmotionState.NEUTRAL).value
                state, confidence = self._stabilize_state(state, confidence)
                return {
                    "success": True,
                    "dominant_emotion": state,
                    "confidence": round(float(confidence), 3),
                }

            return self.emotion_model(frame)
        except Exception as exc:
            logger.error("Emotion detection error: %s", exc)
            return {"success": False}

import cv2
import numpy as np
import mediapipe as mp
import logging
from collections import defaultdict
from enum import Enum

logger = logging.getLogger(__name__)

class EmotionState(Enum):
    CONFIDENT = "confident"
    NEUTRAL = "neutral"
    NERVOUS = "nervous"

class FacialEmotionAnalyzer:
    def __init__(self, model_type="emotion", confidence_threshold=0.6):
        self.confidence_threshold = confidence_threshold
        self.face_detection = mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5
        )
        self.emotion_model = self._load_emotion_model()
        self.emotion_to_state = {
            'happy': EmotionState.CONFIDENT,
            'neutral': EmotionState.NEUTRAL,
            'sad': EmotionState.NERVOUS,
            'angry': EmotionState.NERVOUS,
            'surprise': EmotionState.CONFIDENT,
            'fear': EmotionState.NERVOUS
        }
    
    def _load_emotion_model(self):
        """Try to load FER, else DeepFace, else fallback."""
        try:
            from fer import FER
            return FER()
        except ImportError:
            try:
                from deepface import DeepFace
                return DeepFace
            except ImportError:
                logger.warning("No emotion model found. Using simple fallback.")
                return self._simple_emotion_detector
    
    def _simple_emotion_detector(self, face_image):
        return {'dominant_emotion': 'neutral', 'emotions': {'neutral': 1.0}}
    
    def process_frame(self, frame):
        """
        Returns dict with keys: 'success', 'dominant_emotion' (string), 'confidence' (float)
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb)
        if not results.detections:
            return {'success': False}
        
        # Use first face
        detection = results.detections[0]
        bbox = detection.location_data.relative_bounding_box
        h, w, _ = frame.shape
        x = int(bbox.xmin * w)
        y = int(bbox.ymin * h)
        width = int(bbox.width * w)
        height = int(bbox.height * h)
        x1 = max(0, x - 20)
        y1 = max(0, y - 20)
        x2 = min(w, x + width + 20)
        y2 = min(h, y + height + 20)
        face_roi = frame[y1:y2, x1:x2]
        if face_roi.size == 0:
            return {'success': False}
        
        try:
            if hasattr(self.emotion_model, 'detect_emotions'):  # FER
                result = self.emotion_model.detect_emotions(face_roi)
                if result:
                    emotions = result[0]['emotions']
                    dominant = max(emotions, key=emotions.get)
                    confidence = emotions[dominant]
                else:
                    return {'success': False}
            elif hasattr(self.emotion_model, 'analyze'):  # DeepFace
                result = self.emotion_model.analyze(face_roi, actions=['emotion'], enforce_detection=False)
                if result:
                    dominant = result[0]['dominant_emotion']
                    confidence = result[0]['emotion'][dominant]
                else:
                    return {'success': False}
            else:  # fallback
                result = self.emotion_model(face_roi)
                dominant = result.get('dominant_emotion', 'neutral')
                confidence = result.get('emotions', {}).get(dominant, 0.5)
        except Exception as e:
            logger.error(f"Emotion detection error: {e}")
            return {'success': False}
        
        # Map to our state
        state = self.emotion_to_state.get(dominant.lower(), EmotionState.NEUTRAL).value
        return {
            'success': True,
            'dominant_emotion': state,
            'confidence': confidence
        }
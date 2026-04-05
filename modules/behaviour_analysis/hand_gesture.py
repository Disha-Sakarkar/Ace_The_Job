import cv2
import mediapipe as mp

class HandGestureDetector:
    def __init__(self):
        self.hands = mp.solutions.hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )
        self.last_gesture = None  # Store the last detected gesture
    
    def process_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        
        if not results.multi_hand_landmarks:
            # No hand detected, reset last_gesture so that when hand reappears, we count again
            self.last_gesture = None
            return None
        
        # Use first hand
        hand = results.multi_hand_landmarks[0]
        landmarks = hand.landmark
        
        finger_tips = [8, 12, 16, 20]
        finger_open = 0
        for tip in finger_tips:
            if landmarks[tip].y < landmarks[tip - 2].y:
                finger_open += 1
        
        if finger_open == 0:
            current_gesture = "Fist"
        elif finger_open == 4:
            current_gesture = "Open Palm"
        else:
            current_gesture = "Partial"
        
        # Only return if the gesture changed
        if current_gesture != self.last_gesture:
            self.last_gesture = current_gesture
            return current_gesture
        else:
            return None
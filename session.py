import time
import json
from collections import defaultdict

class SessionData:
    def __init__(self, session_id=None):
        self.session_id = session_id or time.strftime("%Y%m%d_%H%M%S")
        self.start_time = time.time()
        self.end_time = None
        
        # Proctoring data
        self.prohibited_objects = {}          # object_name -> first_seen_time
        self.absence_events = []               # list of incident dicts
        self.multiple_faces_events = []        # list of incident dicts
        self.gaze_away_events = []              # list of incident dicts
        self.head_pose_samples = []             # list of (timestamp, yaw, pitch)
        self.looking_away_frames = 0            # count of frames where looking_away was True
        self.screen_events = []                  # list of event dicts from screen monitor
        self._active_absence_start = None
        self._active_multiple_faces_start = None
        self.warning_counts = {
            "gaze_away": 0
        }
        self.disqualification = {
            "disqualified": False,
            "reason": None,
            "timestamp": None
        }
        
        # Behavior data
        self.emotion_samples = []                # list of (timestamp, emotion, confidence)
        self.gesture_samples = []                 # list of (timestamp, gesture)
        
    def add_prohibited_object(self, obj_name):
        if obj_name not in self.prohibited_objects:
            self.prohibited_objects[obj_name] = round(time.time() - self.start_time, 2)
    
    def _now_offset(self):
        return time.time() - self.start_time

    def _close_incident(self, start_time):
        end_time = self._now_offset()
        return {
            "start_time": round(start_time, 2),
            "end_time": round(end_time, 2),
            "duration_seconds": round(max(0.0, end_time - start_time), 2)
        }

    def start_absence(self):
        if self._active_absence_start is None:
            self._active_absence_start = self._now_offset()
    
    def end_absence(self):
        if self._active_absence_start is not None:
            self.absence_events.append(self._close_incident(self._active_absence_start))
            self._active_absence_start = None
    
    def start_multiple_faces(self):
        if self._active_multiple_faces_start is None:
            self._active_multiple_faces_start = self._now_offset()
    
    def end_multiple_faces(self):
        if self._active_multiple_faces_start is not None:
            self.multiple_faces_events.append(self._close_incident(self._active_multiple_faces_start))
            self._active_multiple_faces_start = None
    
    def add_gaze_away(self, direction, yaw, pitch):
        timestamp = self._now_offset()
        primary_axis = "yaw" if abs(yaw) >= abs(pitch) else "pitch"
        primary_degrees = abs(yaw) if primary_axis == "yaw" else abs(pitch)
        self.gaze_away_events.append({
            "timestamp": round(timestamp, 2),
            "direction": direction,
            "yaw_degrees": round(yaw, 2),
            "pitch_degrees": round(pitch, 2),
            "primary_axis": primary_axis,
            "primary_degrees": round(primary_degrees, 2)
        })
    
    def add_head_pose(self, yaw, pitch, looking_away):
        """Add a head pose sample and track if looking away."""
        self.head_pose_samples.append((time.time() - self.start_time, yaw, pitch))
        if looking_away:
            self.looking_away_frames += 1
    
    def add_screen_event(self, event):
        self.screen_events.append(event)
    
    def add_emotion(self, emotion, confidence):
        self.emotion_samples.append((time.time() - self.start_time, emotion, confidence))
    
    def add_gesture(self, gesture):
        self.gesture_samples.append((time.time() - self.start_time, gesture))

    def add_warning(self, warning_type):
        self.warning_counts[warning_type] = self.warning_counts.get(warning_type, 0) + 1

    def set_disqualification(self, reason):
        if not self.disqualification["disqualified"]:
            self.disqualification = {
                "disqualified": True,
                "reason": reason,
                "timestamp": round(self._now_offset(), 2)
            }

    def finalize_active_events(self):
        self.end_absence()
        self.end_multiple_faces()
    
    def generate_report(self):
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        self.finalize_active_events()
        
        # Proctoring summary
        # Correct gaze away percentage: frames looking away / total head pose frames
        total_pose_frames = len(self.head_pose_samples)
        gaze_away_percentage = (self.looking_away_frames / max(total_pose_frames, 1)) * 100
        
        # Behavior summary
        emotion_counts = defaultdict(int)
        emotion_conf_sum = defaultdict(float)
        for _, em, conf in self.emotion_samples:
            emotion_counts[em] += 1
            emotion_conf_sum[em] += conf
        
        total_emotions = len(self.emotion_samples) or 1
        emotion_dist = {em: (count/total_emotions)*100 for em, count in emotion_counts.items()}
        primary_emotion = max(emotion_dist, key=emotion_dist.get) if emotion_dist else "neutral"
        
        gesture_counts = defaultdict(int)
        for _, g in self.gesture_samples:
            gesture_counts[g] += 1
        
        open_palm = gesture_counts.get("Open Palm", 0)
        fist = gesture_counts.get("Fist", 0)
        partial = gesture_counts.get("Partial", 0)
        
        # Gesture feedback
        total_gestures = open_palm + fist + partial
        gesture_feedback = []
        
        if total_gestures == 0:
            gesture_feedback.append(
                "You didn't use any hand gestures during the interview. "
                "Using purposeful hand movements can make you appear more confident and engaging. "
                "Try incorporating natural gestures while speaking."
            )
        else:
            gpm = total_gestures / (duration / 60) if duration > 0 else 0
            
            if gpm < 20:
                gesture_feedback.append(
                    "You used relatively few hand gestures. "
                    "Adding more expressive movements could enhance your communication and convey confidence."
                )
            elif gpm > 60:
                gesture_feedback.append(
                    "You gestured very frequently. While enthusiasm is good, too many movements can be distracting. "
                    "Try to moderate and make your gestures more deliberate."
                )
            else:
                gesture_feedback.append(
                    "You used a good amount of hand gestures. "
                    "Your movement frequency is appropriate for engaging communication."
                )
            
            open_pct = (open_palm / total_gestures) * 100
            fist_pct = (fist / total_gestures) * 100
            partial_pct = (partial / total_gestures) * 100
            
            if open_pct >= 50:
                gesture_feedback.append(
                    "Most of your gestures were open palms, which convey openness and confidence. Well done."
                )
            elif open_pct >= 25:
                gesture_feedback.append(
                    "You used some open palm gestures. Try to increase them for a more confident impression."
                )
            else:
                gesture_feedback.append(
                    "Your gestures were dominated by fists or unclear movements. "
                    "Aim for more open palms to appear approachable."
                )
            
            if fist_pct >= 30:
                gesture_feedback.append(
                    "A significant portion of your gestures were closed fists, which can signal tension. "
                    "Practice relaxing your hands and using open gestures."
                )
            
            if partial_pct >= 40:
                gesture_feedback.append(
                    "Many of your gestures were partial or unclear. "
                    "Work on making your hand movements more deliberate and complete."
                )
        
        raw_confidence = 100 - (fist * 2 + partial * 1)
        confidence_score = max(0, min(100, raw_confidence))
        total_absence_duration = round(sum(event["duration_seconds"] for event in self.absence_events), 2)
        total_multiple_faces_duration = round(sum(event["duration_seconds"] for event in self.multiple_faces_events), 2)
        gaze_direction_counts = defaultdict(int)
        for event in self.gaze_away_events:
            gaze_direction_counts[event["direction"]] += 1
        
        report = {
            "session_id": self.session_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_seconds": round(duration, 2),
            "proctoring": {
                "prohibited_objects": {
                    "count": len(self.prohibited_objects),
                    "objects_seen": sorted(self.prohibited_objects.keys()),
                    "first_seen_timestamps": self.prohibited_objects
                },
                "absence_events": {
                    "count": len(self.absence_events),
                    "total_duration_seconds": total_absence_duration,
                    "incidents": self.absence_events
                },
                "multiple_faces_events": {
                    "count": len(self.multiple_faces_events),
                    "total_duration_seconds": total_multiple_faces_duration,
                    "incidents": self.multiple_faces_events
                },
                "gaze_away_events": len(self.gaze_away_events),
                "gaze_away_details": {
                    "count": len(self.gaze_away_events),
                    "warnings_shown": self.warning_counts.get("gaze_away", 0),
                    "direction_counts": dict(gaze_direction_counts),
                    "incidents": self.gaze_away_events
                },
                "gaze_away_percentage": round(gaze_away_percentage, 2),
                "head_pose_summary": {
                    "avg_yaw": round(sum(y for _, y, _ in self.head_pose_samples)/max(total_pose_frames,1), 2),
                    "avg_pitch": round(sum(p for _, _, p in self.head_pose_samples)/max(total_pose_frames,1), 2)
                },
                "screen_events": self.screen_events,
                "disqualification": self.disqualification
            },
            "behavior": {
                "emotion_distribution": {k: round(v,2) for k,v in emotion_dist.items()},
                "primary_emotion": primary_emotion,
                "confidence_score": round(confidence_score, 2),
                "hand_gestures": {
                    "open_palm": open_palm,
                    "fist": fist,
                    "partial": partial
                },
                "gesture_feedback": " ".join(gesture_feedback) if gesture_feedback else "No clear gesture pattern."
            }
        }
        return report
    
    def save_json(self, filename=None):
        if filename is None:
            filename = f"interview_{self.session_id}.json"
        with open(filename, 'w') as f:
            json.dump(self.generate_report(), f, indent=2)
        print(f"Report saved to {filename}")

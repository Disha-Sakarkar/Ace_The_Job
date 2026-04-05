import cv2
from modules.cheating_detection.electronic_object import ElectronicObjectDetector
from modules.cheating_detection.person_detection import MultiplePersonDetector
from modules.behaviour_analysis.emotion_detection import FacialEmotionAnalyzer
from modules.behaviour_analysis.hand_gesture import HandGestureDetector
from modules.cheating_detection.gaze_detection import RobustGazeTracker
from modules.cheating_detection.screen_monitor import ScreenMonitor
from session import SessionData


def main():
    max_gaze_warnings = 5
    max_gaze_violations = 5
    warning_display_frames = 90

    print("Initializing modules...")
    session = SessionData()
    modules = {}

    try:
        modules["object"] = ElectronicObjectDetector()
        print("Electronic object detector ready")
    except Exception as e:
        print(f"Object detector failed: {e}")

    try:
        modules["person"] = MultiplePersonDetector(min_detection_confidence=0.4, absence_threshold=3)
        print("Multiple person detector ready")
    except Exception as e:
        print(f"Person detector failed: {e}")

    try:
        modules["emotion"] = FacialEmotionAnalyzer()
        print("Facial emotion analyzer ready")
    except Exception as e:
        print(f"Emotion analyzer failed: {e}")

    try:
        modules["gesture"] = HandGestureDetector()
        print("Hand gesture detector ready")
    except Exception as e:
        print(f"Gesture detector failed: {e}")

    try:
        modules["gaze"] = RobustGazeTracker(yaw_threshold=30, pitch_threshold=20)
        print("Robust gaze tracker ready")
    except Exception as e:
        print(f"Gaze tracker failed: {e}")

    screen_monitor = ScreenMonitor(session)
    screen_monitor.start()
    print("Screen monitor started")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open webcam")
        return

    print("\n=== Interview Started ===")
    print("Press 'q' to stop and save report.\n")

    frame_count = 0
    last_gaze_away = False
    last_absence = False
    last_multiple_faces = False
    gaze_warning_frames_left = 0
    gaze_warning_message = ""
    disqualification_message = ""
    interview_terminated = False
    person_res = {"absence": False, "multiple_faces": False, "face_count": 0}

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            if "object" in modules:
                try:
                    dets = modules["object"].process_frame(frame)
                    for d in dets:
                        session.add_prohibited_object(d["name"])
                except Exception as e:
                    print(f"Object error: {e}")

            if "person" in modules:
                try:
                    person_res = modules["person"].process_frame(frame)
                    if person_res["absence"] and not last_absence:
                        session.start_absence()
                    elif not person_res["absence"] and last_absence:
                        session.end_absence()

                    if person_res["multiple_faces"] and not last_multiple_faces:
                        session.start_multiple_faces()
                    elif not person_res["multiple_faces"] and last_multiple_faces:
                        session.end_multiple_faces()

                    last_absence = person_res["absence"]
                    last_multiple_faces = person_res["multiple_faces"]
                except Exception as e:
                    print(f"Person error: {e}")

            if "emotion" in modules:
                try:
                    emo_res = modules["emotion"].process_frame(frame)
                    if emo_res.get("success"):
                        session.add_emotion(emo_res["dominant_emotion"], emo_res["confidence"])
                except Exception as e:
                    print(f"Emotion error: {e}")

            if "gesture" in modules:
                try:
                    gest = modules["gesture"].process_frame(frame)
                    if gest:
                        session.add_gesture(gest)
                except Exception as e:
                    print(f"Gesture error: {e}")

            if "gaze" in modules:
                try:
                    gaze_res = modules["gaze"].process_frame(frame)
                    if gaze_res.get("success"):
                        session.add_head_pose(gaze_res["yaw"], gaze_res["pitch"], gaze_res["looking_away"])

                        if gaze_res["looking_away"] and not last_gaze_away:
                            session.add_gaze_away(
                                gaze_res["gaze_direction"],
                                gaze_res["yaw"],
                                gaze_res["pitch"]
                            )

                            gaze_incident_count = len(session.gaze_away_events)
                            if gaze_incident_count <= max_gaze_warnings:
                                session.add_warning("gaze_away")
                                gaze_warning_message = (
                                    f"Warning {gaze_incident_count}/{max_gaze_warnings}: "
                                    f"looking {gaze_res['gaze_direction']} "
                                    f"(yaw {gaze_res['yaw']:.1f}, pitch {gaze_res['pitch']:.1f})"
                                )
                                gaze_warning_frames_left = warning_display_frames

                            if gaze_incident_count > max_gaze_violations:
                                disqualification_message = "Interview terminated due to gaze violations"
                                session.set_disqualification(
                                    f"Gaze-away violations exceeded {max_gaze_violations}"
                                )
                                interview_terminated = True

                        last_gaze_away = gaze_res["looking_away"]
                except Exception as e:
                    print(f"Gaze error: {e}")

            if person_res.get("absence"):
                cv2.putText(frame, "CANDIDATE ABSENT", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            if person_res.get("multiple_faces"):
                cv2.putText(frame, "MULTIPLE FACES DETECTED", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            if gaze_warning_frames_left > 0:
                cv2.putText(frame, gaze_warning_message, (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                gaze_warning_frames_left -= 1
            if interview_terminated:
                cv2.putText(frame, disqualification_message, (10, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.putText(frame, "Interview in progress...", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Interview Proctor", frame)

            if interview_terminated:
                cv2.waitKey(2000)
                break

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        if last_absence:
            session.end_absence()
        if last_multiple_faces:
            session.end_multiple_faces()
        cap.release()
        cv2.destroyAllWindows()
        screen_monitor.stop()
        screen_monitor.join()
        print("\nGenerating interview report...")
        session.save_json()
        print("Done.")


if __name__ == "__main__":
    main()

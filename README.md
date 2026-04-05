# Ace The Job

AI-powered interview proctoring and behavior analysis system built with Python, OpenCV, MediaPipe, and YOLO.

## Features

- Candidate presence detection with incident-based absence logging
- Multiple-face detection with false-positive reduction for rotated faces
- Gaze-away tracking with direction, angle estimates, warnings, and disqualification rules
- Electronic object detection with one-time object logging per interview
- Screen monitoring for suspicious window changes, browser changes, and screen-size changes
- Emotion analysis and hand-gesture analysis
- Hand gesture classification for open palm, fist, and partial gestures
- JSON report generation for each interview session

## Project Structure

```text
.
|-- main.py
|-- session.py
|-- requirements.txt
|-- yolov8n.pt
`-- modules/
    |-- behaviour_analysis/
    |   |-- emotion_detection.py
    |   `-- hand_gesture.py
    `-- cheating_detection/
        |-- electronic_object.py
        |-- gaze_detection.py
        |-- person_detection.py
        `-- screen_monitor.py
```

## How It Works

The application opens the webcam, runs multiple detectors in parallel, shows real-time warnings on screen, and writes a JSON report at the end of the session.

Current proctoring behavior includes:

- `absence_events`: logs real absence incidents with start time, end time, and duration
- `multiple_faces_events`: logs real multiple-face incidents instead of frame counts
- `prohibited_objects`: stores only which prohibited objects were seen during the interview
- `gaze_away_details`: stores look-away incidents with direction and angle information
- `Gaze policy`: warning for the first 5 gaze-away violations, then disqualification if the count goes beyond 5
- `hand_gestures`: tracks open palm, fist, and partial hand gestures and includes summary feedback

## Installation

1. Create and activate a Python virtual environment.
2. Install the dependencies:

```bash
pip install -r requirements.txt
```

3. Make sure `yolov8n.pt` is present in the project root.

## Run

```bash
python main.py
```

Press `q` to stop the session and save the report.

## Output

At the end of the interview, the app generates a JSON report containing:

- Prohibited objects seen
- Absence incidents
- Multiple-face incidents
- Gaze-away incidents and warnings
- Screen-monitoring events
- Emotion distribution
- Hand gesture counts and feedback, including open palm, fist, and partial gestures

## Privacy Notes

This repository ignores local/generated data such as:

- interview JSON reports
- local monitor screenshots/logs
- virtual environment files
- cache files

That helps avoid pushing personal interview data to GitHub by mistake.

## Tech Stack

- Python
- OpenCV
- MediaPipe
- Ultralytics YOLO
- NumPy
- psutil
- pyautogui
- mss

## Future Improvements

- Better packaging and setup instructions
- Model/config separation for easier tuning
- UI improvements for warnings and session status
- Persistent database or dashboard for reports

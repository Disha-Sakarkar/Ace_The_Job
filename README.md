# Ace The Job

AI-powered interview proctoring and behavior analysis system built with Python, OpenCV, MediaPipe, YOLO, and FER.

## Features

- Candidate presence detection with incident-based absence logging
- Multiple-face detection with false-positive reduction for rotated faces
- Gaze-away tracking with direction, angle estimates, warnings, and disqualification rules
- Electronic object detection with one-time object logging per interview
- Screen monitoring for suspicious window changes, browser changes, and screen-size changes
- Emotion analysis with FER-backed facial expression classification and a fallback heuristic path
- Hand-gesture analysis
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

The application opens the webcam, runs the webcam analysis modules sequentially on each frame, runs screen monitoring in a separate background thread, shows real-time warnings on screen, and writes a JSON report at the end of the session.

Current proctoring behavior includes:

- `absence_events`: logs real absence incidents with start time, end time, and duration
- `multiple_faces_events`: logs real multiple-face incidents instead of frame counts
- `prohibited_objects`: stores only which prohibited objects were seen during the interview
- `gaze_away_details`: stores look-away incidents with direction and angle information
- `Gaze policy`: warning for the first 5 gaze-away violations, then disqualification if the count goes beyond 5
- `emotion_distribution`: stores emotion samples mapped into `confident`, `neutral`, and `nervous`
- `hand_gestures`: tracks open palm, fist, and partial hand gestures and includes summary feedback

## Emotion Detection

The emotion pipeline is implemented in [`modules/behaviour_analysis/emotion_detection.py`](modules/behaviour_analysis/emotion_detection.py).

Current behavior:

- The module first tries to load `FER` from the `fer` package using `FER(use_tflite=True)`.
- FER performs face detection and then classifies the detected face into the standard expression labels `angry`, `disgust`, `fear`, `happy`, `sad`, `surprise`, and `neutral`.
- The project then maps those raw FER labels into interview-friendly behavior states:
  - `happy`, `surprise` -> `confident`
  - `neutral` -> `neutral`
  - `sad`, `angry`, `fear`, `disgust` -> `nervous`
- If FER cannot be loaded, the module falls back to `DeepFace` if available.
- If neither FER nor DeepFace is available, the module falls back to a local MediaPipe landmark-based heuristic so the rest of the project can continue running.

Notes:

- The main application consumes only the normalized output `success`, `dominant_emotion`, and `confidence`.
- Emotion samples are recorded in the session report under `behavior.emotion_distribution` and `behavior.primary_emotion`.
- The report field `confidence_score` is currently derived from hand gestures, not from facial emotion confidence.
## Installation

1. Create and activate a Python virtual environment.
2. Install the dependencies:

```bash
pip install -r requirements.txt
```

3. Make sure `yolov8n.pt` is present in the project root.
4. On Linux, install the required system tools separately if you want screen monitoring support:

```bash
sudo apt-get install xdotool wmctrl
```

## Run

Activate the environment in PowerShell:

```powershell
.\interview_env\Scripts\Activate.ps1
```

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
- FER
- TensorFlow
- NumPy
- psutil
- pyautogui
- mss

## Future Improvements

- Better packaging and setup instructions
- Model/config separation for easier tuning
- UI improvements for warnings and session status
- Persistent database or dashboard for reports

import cv2
import mediapipe as mp
import time

class MultiplePersonDetector:
    def __init__(
        self,
        min_detection_confidence=0.4,
        absence_threshold=3,
        multiple_faces_threshold=2,
        min_face_score=0.45,
        overlap_iou_threshold=0.3,
        center_distance_threshold=0.08,
        absence_seconds_threshold=1.5,
        multiple_faces_seconds_threshold=1.0
    ):
        """
        Args:
            min_detection_confidence: Lower threshold to detect faces more easily.
            absence_threshold: Number of consecutive frames with no face to count as absence.
            multiple_faces_threshold: Consecutive frames needed before confirming multiple faces.
            min_face_score: Ignore weak face detections that are likely false positives.
            overlap_iou_threshold: Merge overlapping detections that likely belong to the same face.
            center_distance_threshold: Merge detections whose centers are very close.
            absence_seconds_threshold: Require missing face for this many seconds before absence.
            multiple_faces_seconds_threshold: Require multiple faces for this many seconds before violation.
        """
        self.face_detection = mp.solutions.face_detection.FaceDetection(
            model_selection=0,  # 0 for short-range (better for close-up)
            min_detection_confidence=min_detection_confidence
        )
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=2,
            refine_landmarks=False,
            min_detection_confidence=0.35,
            min_tracking_confidence=0.35
        )
        self.absence_threshold = absence_threshold
        self.multiple_faces_threshold = multiple_faces_threshold
        self.min_face_score = min_face_score
        self.overlap_iou_threshold = overlap_iou_threshold
        self.center_distance_threshold = center_distance_threshold
        self.absence_seconds_threshold = absence_seconds_threshold
        self.multiple_faces_seconds_threshold = multiple_faces_seconds_threshold
        self.no_face_counter = 0  # consecutive frames without a face
        self.multiple_face_counter = 0  # consecutive frames with confirmed distinct faces
        self.absence_start_time = None
        self.multiple_faces_start_time = None

    def _get_face_boxes(self, detections, frame_shape):
        h, w = frame_shape[:2]
        boxes = []

        for detection in detections:
            scores = detection.score if detection.score else []
            score = float(scores[0]) if scores else 0.0
            if score < self.min_face_score:
                continue

            bbox = detection.location_data.relative_bounding_box
            x1 = max(0.0, bbox.xmin)
            y1 = max(0.0, bbox.ymin)
            x2 = min(1.0, bbox.xmin + bbox.width)
            y2 = min(1.0, bbox.ymin + bbox.height)

            if x2 <= x1 or y2 <= y1:
                continue

            area = (x2 - x1) * (y2 - y1)
            if area < 0.005:
                continue

            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            boxes.append({
                'score': score,
                'x1': x1,
                'y1': y1,
                'x2': x2,
                'y2': y2,
                'cx': cx,
                'cy': cy,
                'w': x2 - x1,
                'h': y2 - y1,
                'frame_w': w,
                'frame_h': h
            })

        return boxes

    def _iou(self, box_a, box_b):
        inter_x1 = max(box_a['x1'], box_b['x1'])
        inter_y1 = max(box_a['y1'], box_b['y1'])
        inter_x2 = min(box_a['x2'], box_b['x2'])
        inter_y2 = min(box_a['y2'], box_b['y2'])

        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        area_a = box_a['w'] * box_a['h']
        area_b = box_b['w'] * box_b['h']
        union_area = area_a + area_b - inter_area

        if union_area <= 0:
            return 0.0

        return inter_area / union_area

    def _is_same_face(self, box_a, box_b):
        if self._iou(box_a, box_b) >= self.overlap_iou_threshold:
            return True

        center_distance = ((box_a['cx'] - box_b['cx']) ** 2 + (box_a['cy'] - box_b['cy']) ** 2) ** 0.5
        mean_face_size = max((box_a['w'] + box_b['w'] + box_a['h'] + box_b['h']) / 4.0, 1e-6)

        return center_distance < min(self.center_distance_threshold, mean_face_size * 0.45)

    def _count_distinct_faces(self, detections, frame_shape):
        boxes = sorted(
            self._get_face_boxes(detections, frame_shape),
            key=lambda box: box['score'],
            reverse=True
        )

        distinct_faces = []
        for box in boxes:
            if any(self._is_same_face(box, kept_box) for kept_box in distinct_faces):
                continue
            distinct_faces.append(box)

        return len(distinct_faces)
    
    def process_frame(self, frame):
        now = time.monotonic()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detection_results = self.face_detection.process(rgb)
        mesh_results = self.face_mesh.process(rgb)

        detection_face_count = 0
        if detection_results.detections:
            detection_face_count = self._count_distinct_faces(detection_results.detections, frame.shape)

        mesh_face_count = 0
        if mesh_results.multi_face_landmarks:
            mesh_face_count = len(mesh_results.multi_face_landmarks)

        # For presence, trust either pipeline if it sees at least one face.
        visible_face_count = max(detection_face_count, mesh_face_count)

        # For multiple faces, require stronger confirmation to suppress rotation artifacts.
        confirmed_multiple_faces = mesh_face_count > 1 or detection_face_count > 1

        # Update consecutive no-face counter
        if visible_face_count == 0:
            self.no_face_counter += 1
            if self.absence_start_time is None:
                self.absence_start_time = now
        else:
            self.no_face_counter = 0
            self.absence_start_time = None

        # Absence is only true if no face is seen consistently and long enough.
        absence_duration = 0.0 if self.absence_start_time is None else now - self.absence_start_time
        absence = (
            visible_face_count == 0 and
            self.no_face_counter >= self.absence_threshold and
            absence_duration >= self.absence_seconds_threshold
        )

        if confirmed_multiple_faces:
            self.multiple_face_counter += 1
            if self.multiple_faces_start_time is None:
                self.multiple_faces_start_time = now
        else:
            self.multiple_face_counter = 0
            self.multiple_faces_start_time = None

        multiple_faces_duration = (
            0.0 if self.multiple_faces_start_time is None else now - self.multiple_faces_start_time
        )
        multiple_faces = (
            confirmed_multiple_faces and
            self.multiple_face_counter >= self.multiple_faces_threshold and
            multiple_faces_duration >= self.multiple_faces_seconds_threshold
        )
        
        return {
            'face_count': visible_face_count,
            'absence': absence,
            'multiple_faces': multiple_faces,
            'detection_face_count': detection_face_count,
            'mesh_face_count': mesh_face_count
        }

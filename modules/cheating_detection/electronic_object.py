from ultralytics import YOLO

class ElectronicObjectDetector:
    def __init__(self, model_path="yolov8n.pt", conf_threshold=0.35):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        # Prohibited tokens (same as in device_detection_fixed.py)
        self.prohibited_tokens = [
            "phone", "cell", "mobile", "smartphone",
            "laptop", "tablet",
            "calculator",
            "earbud", "earphone", "headphone", "headset",
            "smartwatch",
            "camera",
            "usb", "flash", "pendrive",
            "speaker",
            "pager",
            "book", "pen", "pencil"
        ]
    
    def is_prohibited(self, class_name):
        name = class_name.lower()
        return any(token in name for token in self.prohibited_tokens)
    
    def process_frame(self, frame):
        results = self.model(frame, imgsz=640, verbose=False)[0]
        detections = []
        if results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy().astype(int)
            confs = results.boxes.conf.cpu().numpy()
            names = results.names
            
            for box, cls, conf in zip(boxes, classes, confs):
                if conf < self.conf_threshold:
                    continue
                class_name = names[cls]
                if self.is_prohibited(class_name):
                    detections.append({
                        'name': class_name,
                        'confidence': float(conf),
                        'bbox': box.tolist()
                    })
        return detections
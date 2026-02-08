from ultralytics import YOLO
import numpy as np
import os

class PPEDetector:
    def __init__(self,model_path:str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model weights not found {model_path}")
    
        try:
            self.model = YOLO(model_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load the model: {e}")
        
    def detect(self,image_path:str):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found")
            
        try:
            results = self.model(image_path,conf = 0.20)
        except Exception as e:
            raise RuntimeError(f"Inference failed{e}")

        detection = []
        for r in results:
            for box in r.boxes:
                    
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].tolist()
                detection.append(
                    {
                        "class_id":cls_id,
                        "confidence":conf,
                        "bbox":xyxy
                        }
                    )
        return detection


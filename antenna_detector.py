import torch
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as transforms
from ultralytics import YOLO
import os

class AntennaDetector:
    def __init__(self):
        # Load YOLOv8 model
        model_path = os.path.join('UTD-Models-Videos', 'runs', 'detect', 'train4', 'weights', 'best.pt')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        self.model = YOLO(model_path)
        self.classes = ['GSM Antenna', 'Microwave Antenna', 'antenna', 'Lattice Tower', 'M Type Tower']
        
    def detect_antennas(self, image):
        # Ensure image is in RGB format
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        # Run inference
        results = self.model(image)
        
        # Process results
        detections = []
        scores = []
        antenna_count = 0
        
        # Process each detection
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                # Get confidence
                conf = float(box.conf[0].cpu().numpy())
                # Get class
                cls = int(box.cls[0].cpu().numpy())
                class_name = self.classes[cls]
                
                # Only count antenna detections
                if 'antenna' in class_name.lower():
                    antenna_count += 1
                    detections.append([int(x1), int(y1), int(x2), int(y2)])
                    scores.append(conf)
        
        return detections, scores, antenna_count
        
    def draw_boxes(self, image, boxes, scores):
        # Convert PIL Image to OpenCV format if needed
        if isinstance(image, Image.Image):
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        else:
            image = image.copy()
            
        antenna_count = len(boxes)
        
        # Draw total antenna count at the top of the image
        cv2.putText(image, f'Total Antennas: {antenna_count}', 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = box
            # Draw rectangle
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Add score text
            text = f'Antenna: {score:.2f}'
            cv2.putText(image, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert back to RGB for Gradio 
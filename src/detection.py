"""
Tower Detection Module

This module handles the detection of cell towers and their components
using YOLOv8 object detection models.
"""

import os
import cv2
import numpy as np
from typing import List, Dict, Tuple, Any, Optional, Union
import time
from pathlib import Path

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

class TowerDetector:
    """
    Detector class for identifying towers and tower components in images and videos.
    Supports both YOLOv8 and OpenCV DNN-based YOLO models.
    """
    
    def __init__(self, 
                 model_path: str,
                 confidence_threshold: float = 0.3,
                 nms_threshold: float = 0.45,
                 use_cuda: bool = True):
        """
        Initialize the tower detector with model parameters.
        
        Args:
            model_path: Path to the YOLOv8 model or YOLOv3/v4 weights
            confidence_threshold: Minimum confidence for detection
            nms_threshold: Non-maximum suppression threshold
            use_cuda: Whether to use CUDA acceleration
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.use_cuda = use_cuda and self._is_cuda_available()
        
        # Load the model
        self._load_model()
        
        # Tower-related class names (based on your custom dataset)
        self.tower_classes = [
            "lattice_tower", "monopole_tower", "guyed_tower", 
            "cell_antenna", "microwave_antenna", "base_station",
            "remote_radio_head", "equipment_cabinet", "dish_antenna"
        ]
        
    def _is_cuda_available(self) -> bool:
        """Check if CUDA is available for acceleration"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def _load_model(self):
        """Load the appropriate model based on file extension"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
        file_ext = os.path.splitext(self.model_path)[1].lower()
        
        if file_ext in ['.pt', '.pth'] and YOLO_AVAILABLE:
            # YOLOv8 from Ultralytics
            self.model = YOLO(self.model_path)
            self.model_type = "yolov8"
            print(f"Loaded YOLOv8 model from {self.model_path}")
        else:
            # Fall back to OpenCV DNN implementation for older YOLO versions
            self._load_opencv_model()
    
    def _load_opencv_model(self):
        """Load YOLO model using OpenCV DNN"""
        self.model_type = "opencv"
        
        # For YOLO format, we need both weights and config files
        if self.model_path.endswith('.weights'):
            # Extract the base directory and try to find a matching .cfg file
            base_dir = os.path.dirname(self.model_path)
            base_name = os.path.splitext(os.path.basename(self.model_path))[0]
            config_path = os.path.join(base_dir, 'cfg', f"{base_name}.cfg")
            
            if not os.path.exists(config_path):
                # Try alternative locations
                config_path = os.path.join(base_dir, f"{base_name}.cfg")
                if not os.path.exists(config_path):
                    raise FileNotFoundError(f"Could not find config file for {self.model_path}")
            
            # Load the names file if available
            names_path = os.path.join(base_dir, 'data', 'coco.names')
            if os.path.exists(names_path):
                with open(names_path, 'r') as f:
                    self.classes = [line.strip() for line in f.readlines()]
            else:
                print(f"Warning: Names file not found at {names_path}")
                self.classes = [f"class_{i}" for i in range(80)]  # Default to 80 COCO classes
                
            # Load network
            self.net = cv2.dnn.readNetFromDarknet(config_path, self.model_path)
            
            # Set backend and target
            if self.use_cuda:
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            else:
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                
            # Get output layer names
            layer_names = self.net.getLayerNames()
            self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
            
            print(f"Loaded YOLO model via OpenCV DNN from {self.model_path} and {config_path}")
        else:
            raise ValueError("Unsupported model format. Expected .pt (YOLOv8) or .weights (YOLOv3/v4)")
    
    def detect(self, 
               image: np.ndarray, 
               draw: bool = True) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """
        Detect towers and components in an image.
        
        Args:
            image: Input image as numpy array (BGR format)
            draw: Whether to draw detections on the output image
            
        Returns:
            Tuple of (detections, output_image)
        """
        if self.model_type == "yolov8":
            return self._detect_yolov8(image, draw)
        else:
            return self._detect_opencv(image, draw)
    
    def _detect_yolov8(self, 
                      image: np.ndarray, 
                      draw: bool = True) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """Detect using YOLOv8 model from Ultralytics"""
        # Make copy for drawing
        output_img = image.copy() if draw else image
        
        # Run inference
        results = self.model(image, conf=self.confidence_threshold)
        
        # Parse detections
        detections = []
        
        # Process results (assuming Ultralytics YOLO output format)
        for result in results:
            boxes = result.boxes
            
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = box.conf[0].cpu().numpy()
                class_id = int(box.cls[0].cpu().numpy())
                
                if hasattr(result, 'names') and result.names is not None:
                    class_name = result.names[class_id]
                else:
                    class_name = f"class_{class_id}"
                
                # Create detection entry
                detection = {
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': float(confidence),
                    'class_id': class_id,
                    'class_name': class_name
                }
                detections.append(detection)
                
                # Draw if requested
                if draw:
                    color = (0, 255, 0)  # Green by default
                    cv2.rectangle(output_img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    label = f"{class_name}: {confidence:.2f}"
                    y = int(y1) - 10 if int(y1) - 10 > 10 else int(y1) + 20
                    cv2.putText(output_img, label, (int(x1), y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return detections, output_img
    
    def _detect_opencv(self, 
                      image: np.ndarray, 
                      draw: bool = True) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """Detect using OpenCV DNN implementation"""
        height, width = image.shape[:2]
        
        # Make copy for drawing
        output_img = image.copy() if draw else image
        
        # Create a blob from the image
        blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
        
        # Set input and perform forward pass
        self.net.setInput(blob)
        start_time = time.time()
        outs = self.net.forward(self.output_layers)
        inference_time = time.time() - start_time
        
        # Process outputs
        boxes = []
        confidences = []
        class_ids = []
        
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > self.confidence_threshold:
                    # YOLO returns center, width, height
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    # Convert to top-left corner
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        # Apply non-maximum suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, self.nms_threshold)
        
        # Create detection objects
        detections = []
        
        for i in indices:
            if isinstance(i, (list, tuple)):  # OpenCV 4.5.4 and earlier might return a list of lists
                i = i[0]
                
            box = boxes[i]
            x, y, w, h = box
            class_id = class_ids[i]
            confidence = confidences[i]
            
            # Convert to x1, y1, x2, y2 format
            x1, y1, x2, y2 = x, y, x + w, y + h
            
            # Get class name if available
            if hasattr(self, 'classes') and class_id < len(self.classes):
                class_name = self.classes[class_id]
            else:
                class_name = f"class_{class_id}"
            
            # Create detection entry
            detection = {
                'bbox': [x1, y1, x2, y2],
                'confidence': confidence,
                'class_id': int(class_id),
                'class_name': class_name
            }
            detections.append(detection)
            
            # Draw if requested
            if draw:
                color = (0, 255, 0)  # Green by default
                cv2.rectangle(output_img, (x1, y1), (x2, y2), color, 2)
                label = f"{class_name}: {confidence:.2f}"
                y_pos = y1 - 10 if y1 - 10 > 10 else y1 + 20
                cv2.putText(output_img, label, (x1, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return detections, output_img
    
    def is_tower_in_image(self, image: np.ndarray) -> bool:
        """
        Check if there's a tower in the image.
        
        Args:
            image: Input image
            
        Returns:
            Boolean indicating if a tower was detected
        """
        detections, _ = self.detect(image, draw=False)
        
        for detection in detections:
            class_name = detection['class_name'].lower()
            # Check if any tower-related classes are detected
            if any(tower_class in class_name for tower_class in self.tower_classes):
                return True
        
        return False
    
    def filter_tower_detections(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter detections to only include tower-related objects.
        
        Args:
            detections: List of detection dictionaries
            
        Returns:
            List of tower-related detections
        """
        tower_detections = []
        
        for detection in detections:
            class_name = detection['class_name'].lower()
            # Keep only tower-related detections
            if any(tower_class in class_name for tower_class in self.tower_classes):
                tower_detections.append(detection)
        
        return tower_detections
    
    def process_video(self, 
                     video_path: str, 
                     output_path: Optional[str] = None,
                     frame_skip: int = 10,
                     max_frames: Optional[int] = None,
                     output_fps: int = 15) -> Dict[str, Any]:
        """
        Process a video file for tower detection.
        
        Args:
            video_path: Path to input video
            output_path: Path to save output video (None to skip saving)
            frame_skip: Process every Nth frame
            max_frames: Maximum number of frames to process
            output_fps: FPS for output video
            
        Returns:
            Dictionary with processing results
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Setup video writer if output path is specified
        output_writer = None
        if output_path is not None:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            output_writer = cv2.VideoWriter(output_path, fourcc, output_fps, (width, height))
        
        # Process video
        frame_count = 0
        processed_frames = 0
        tower_frames = 0
        frame_results = []
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
                
            # Skip frames for efficiency
            if frame_count % frame_skip != 0:
                frame_count += 1
                continue
            
            # Process frame
            detections, result_frame = self.detect(frame)
            
            # Filter for tower detections
            tower_detections = self.filter_tower_detections(detections)
            
            # Save the processed frame
            if output_writer is not None:
                output_writer.write(result_frame)
                
            # Update counters
            processed_frames += 1
            if tower_detections:
                tower_frames += 1
                
            # Store results
            frame_results.append({
                'frame_idx': frame_count,
                'detections': detections,
                'tower_detections': tower_detections
            })
            
            # Check if we've reached max frames
            if max_frames is not None and processed_frames >= max_frames:
                break
                
            frame_count += 1
        
        # Release resources
        cap.release()
        if output_writer is not None:
            output_writer.release()
            
        # Compile results
        results = {
            'video_path': video_path,
            'output_path': output_path,
            'total_frames': total_frames,
            'processed_frames': processed_frames,
            'tower_frames': tower_frames,
            'frame_results': frame_results,
            'detection_rate': tower_frames / processed_frames if processed_frames > 0 else 0
        }
        
        return results
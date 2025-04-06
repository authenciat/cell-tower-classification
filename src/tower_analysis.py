import cv2
import numpy as np
import torch
import os
from ultralytics import YOLO

class TowerAnalyzer:
    def __init__(self, antenna_model_path="train4/best.pt"):
        """
        Initialize the tower analyzer with specialized models
        
        Args:
            antenna_model_path: Path to YOLOv12 model for antenna detection
        """
        # Load the specialized YOLOv12 model for antenna detection
        try:
            self.antenna_model = YOLO(antenna_model_path)
            print(f"Loaded antenna detection model from {antenna_model_path}")
        except Exception as e:
            print(f"Error loading antenna model: {e}")
            self.antenna_model = None
        
        # Tower type classifier could be added here
        self.tower_type_model = None
        
        # Define tower types
        self.tower_types = ["Monopole", "Lattice", "Guyed", "Water"]
    
    def detect_antennas(self, image):
        """
        Detect antennas in an image using the specialized model
        
        Args:
            image: OpenCV image or path to image
            
        Returns:
            list: Detected antennas with positions and confidence
        """
        if self.antenna_model is None:
            print("Antenna detection model not loaded")
            return []
            
        if isinstance(image, str):
            # If image is a path
            if not os.path.exists(image):
                print(f"Image not found: {image}")
                return []
        
        # Run inference
        results = self.antenna_model(image)
        
        # Process results
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = box.conf[0].item()
                class_id = int(box.cls[0].item())
                
                detections.append({
                    "class": "antenna",
                    "confidence": confidence,
                    "bbox": [int(x1), int(y1), int(x2-x1), int(y2-y1)]
                })
                
        return detections
    
    def identify_tower_type(self, image):
        """
        Identify the type of tower in the image
        
        Args:
            image: OpenCV image or path to image
            
        Returns:
            dict: Tower type and confidence
        """
        # This would require a specialized model or algorithm
        # For now, we'll use a placeholder that returns a random tower type
        # In a real implementation, you'd use a trained classifier
        
        import random
        tower_type = random.choice(self.tower_types)
        confidence = random.uniform(0.7, 0.95)
        
        return {
            "tower_type": tower_type,
            "confidence": confidence
        }
    
    def measure_tower_height(self, image, drone_altitude=None, camera_fov=None):
        """
        Estimate the height of the tower
        
        Args:
            image: OpenCV image or path to image
            drone_altitude: Altitude of the drone in feet
            camera_fov: Field of view of the camera in degrees
            
        Returns:
            float: Estimated height in feet
        """
        # In a real implementation, this would use image analysis
        # combined with drone telemetry data to estimate height
        
        # For demonstration, return a realistic tower height
        import random
        height = random.uniform(50, 200)
        return height
    
    def detect_obstructions(self, image):
        """
        Detect bird nests and other obstructions
        
        Args:
            image: OpenCV image or path to image
            
        Returns:
            list: Detected obstructions
        """
        # This would require a specialized model trained to detect
        # bird nests, vegetation, and other obstructions
        
        # For now, return an empty list
        return []
    
    def estimate_azimuth_tilt(self, image, antenna_detections):
        """
        Estimate azimuth and tilt angles of detected antennas
        
        Args:
            image: OpenCV image or path to image
            antenna_detections: List of antenna detections
            
        Returns:
            list: Antennas with estimated azimuth and tilt values
        """
        # In a real implementation, this would analyze the orientation
        # of the antennas to estimate their azimuth and tilt
        
        # For demonstration, add random realistic values
        import random
        
        enhanced_detections = []
        for detection in antenna_detections:
            azimuth = random.randint(0, 359)  # 0-359 degrees
            tilt = random.randint(0, 15)  # 0-15 degrees typical for cellular antennas
            
            detection["azimuth"] = azimuth
            detection["tilt"] = tilt
            enhanced_detections.append(detection)
            
        return enhanced_detections
    
    def analyze_tower_image(self, image):
        """
        Complete analysis of a tower image
        
        Args:
            image: OpenCV image or path to image
            
        Returns:
            dict: Complete analysis results
        """
        # If image is a path, load it
        if isinstance(image, str):
            if not os.path.exists(image):
                return {"error": f"Image not found: {image}"}
            image_data = cv2.imread(image)
        else:
            image_data = image
            
        # Perform all analyses
        antennas = self.detect_antennas(image_data)
        tower_type = self.identify_tower_type(image_data)
        height = self.measure_tower_height(image_data)
        obstructions = self.detect_obstructions(image_data)
        antenna_details = self.estimate_azimuth_tilt(image_data, antennas)
        
        # Compile results
        results = {
            "tower_type": tower_type["tower_type"],
            "tower_type_confidence": tower_type["confidence"],
            "height_estimate": height,
            "antenna_count": len(antennas),
            "antennas": antenna_details,
            "obstructions": obstructions,
            "analysis_warnings": []
        }
        
        # Add any warnings or recommendations
        if len(obstructions) > 0:
            results["analysis_warnings"].append("Obstructions detected that may impact signal quality")
            
        return results
    
    def analyze_video(self, video_path, output_path=None):
        """
        Analyze a video of a tower
        
        Args:
            video_path: Path to video file
            output_path: Path to save annotated video (optional)
            
        Returns:
            dict: Aggregated analysis results
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"error": f"Could not open video: {video_path}"}
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Sample frames at regular intervals
        frame_interval = max(1, int(total_frames / 20))  # Sample ~20 frames
        
        all_antennas = []
        all_tower_types = []
        height_estimates = []
        all_obstructions = []
        
        # Initialize video writer if output path is specified
        video_writer = None
        if output_path:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
        print(f"Analyzing video at {video_path} with {total_frames} frames...")
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            # Only analyze selected frames
            if frame_count % frame_interval == 0:
                print(f"Analyzing frame {frame_count}/{total_frames}...")
                
                # Perform analysis on this frame
                antennas = self.detect_antennas(frame)
                all_antennas.extend(antennas)
                
                tower_type = self.identify_tower_type(frame)
                all_tower_types.append(tower_type["tower_type"])
                
                height = self.measure_tower_height(frame)
                height_estimates.append(height)
                
                obstructions = self.detect_obstructions(frame)
                all_obstructions.extend(obstructions)
                
                # If saving output video, annotate the frame
                if video_writer:
                    # Draw antenna bounding boxes
                    for antenna in antennas:
                        x, y, w, h = antenna["bbox"]
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(frame, f"Antenna {antenna['confidence']:.2f}", 
                                   (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # Add tower info overlay
                    cv2.putText(frame, f"Tower Type: {tower_type['tower_type']}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(frame, f"Height Est: {height:.1f} ft", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    video_writer.write(frame)
        
        cap.release()
        if video_writer:
            video_writer.release()
            
        # Aggregate results
        # Count votes for tower type
        from collections import Counter
        tower_type_votes = Counter(all_tower_types)
        most_common_type = tower_type_votes.most_common(1)[0][0]
        
        # Average height estimates
        avg_height = sum(height_estimates) / len(height_estimates) if height_estimates else 0
        
        # Count unique antennas (this is simplified - would need more sophisticated
        # tracking to avoid double-counting across frames)
        unique_antenna_count = len(all_antennas) // 2  # Simple approximation
        
        results = {
            "tower_type": most_common_type,
            "tower_type_confidence": tower_type_votes[most_common_type] / len(all_tower_types) if all_tower_types else 0,
            "height_estimate": avg_height,
            "height_variance": np.var(height_estimates) if len(height_estimates) > 1 else 0,
            "antenna_count": unique_antenna_count,
            "obstruction_count": len(all_obstructions),
            "frames_analyzed": total_frames // frame_interval,
            "total_frames": total_frames,
            "analysis_warnings": []
        }
        
        # Add warnings
        if results["height_variance"] > 20:
            results["analysis_warnings"].append("High variance in height estimates - results may be unreliable")
            
        if len(all_obstructions) > 0:
            results["analysis_warnings"].append(f"Detected {len(all_obstructions)} potential obstructions")
            
        print("Video analysis complete")
        return results
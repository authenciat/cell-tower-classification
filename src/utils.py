"""
Utility module for video processing and helper functions
"""

import cv2
import numpy as np
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple, Union, Optional, Iterator
import time
import tempfile
import json
import glob

class VideoProcessor:
    """
    Utility class for processing video files for tower detection
    """
    
    def __init__(self, detector=None, analyzer=None, visualizer=None):
        """
        Initialize the video processor
        
        Args:
            detector: Optional tower detector instance
            analyzer: Optional tower analyzer instance
            visualizer: Optional visualizer instance
        """
        self.detector = detector
        self.analyzer = analyzer
        self.visualizer = visualizer
        
        # Video processing settings
        self.frame_skip = 15  # Process every 15th frame (increased from 5)
        self.max_frames = 10  # Reduced from 25 to 10 frames max
        self.output_fps = 10  # Default output FPS
        
    def extract_frames(self, 
                    video_path: str, 
                    output_dir: str = None,
                    num_frames: int = 10,
                    uniform: bool = True,
                    save_frames: bool = True) -> List[np.ndarray]:
        """
        Extract frames from a video file
        
        Args:
            video_path: Path to the video file
            output_dir: Directory to save extracted frames
            num_frames: Number of frames to extract
            uniform: Whether to extract frames uniformly across the video
            save_frames: Whether to save frames to disk
            
        Returns:
            frames: List of extracted frames
        """
        # Check if video file exists
        if not os.path.exists(video_path):
            print(f"Error: Video file not found: {video_path}")
            return []
            
        # Create output directory if needed
        if save_frames and output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            
        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video: {video_path}")
            return []
            
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
        
        # Determine frame indices to extract
        if uniform:
            # Extract frames uniformly across the video
            if total_frames <= num_frames:
                # If video has fewer frames than requested, use all frames
                frame_indices = list(range(total_frames))
            else:
                # Calculate frame intervals
                interval = total_frames / num_frames
                frame_indices = [int(i * interval) for i in range(num_frames)]
        else:
            # Extract frames from the beginning
            frame_indices = list(range(min(num_frames, total_frames)))
            
        frames = []
        frame_paths = []
        
        # Extract the frames
        for i, frame_idx in enumerate(frame_indices):
            # Set position to the desired frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                print(f"Warning: Could not read frame {frame_idx}")
                continue
                
            frames.append(frame)
            
            # Save frame if requested
            if save_frames and output_dir is not None:
                video_name = os.path.splitext(os.path.basename(video_path))[0]
                frame_path = os.path.join(output_dir, f"{video_name}_frame_{i:03d}.jpg")
                cv2.imwrite(frame_path, frame)
                frame_paths.append(frame_path)
                print(f"Saved frame {i+1}/{len(frame_indices)} (index {frame_idx}) to {frame_path}")
                
        cap.release()
        
        return frames, frame_paths
        
    def process_video(self, 
                    video_path: str, 
                    output_path: str = None, 
                    save_frames: bool = False,
                    frames_dir: str = None,
                    analyze: bool = True) -> Dict[str, Any]:
        """
        Process a video file for tower detection
        
        Args:
            video_path: Path to the video file
            output_path: Path to save the processed video (not used anymore, kept for backward compatibility)
            save_frames: Whether to save extracted frames
            frames_dir: Directory to save frames (if save_frames is True)
            analyze: Whether to perform tower analysis
            
        Returns:
            results: Dictionary with processing results
        """
        if not os.path.exists(video_path):
            return {"error": f"Video file not found: {video_path}"}
            
        if self.detector is None:
            return {"error": "No detector provided"}
            
        # Create frames directory if saving frames
        if save_frames:
            if frames_dir is None:
                frames_dir = tempfile.mkdtemp(prefix="tower_frames_")
            else:
                os.makedirs(frames_dir, exist_ok=True)
                
        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"error": f"Could not open video: {video_path}"}
            
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # No longer initializing video writer
        
        # Process frames
        results = {
            "video_path": video_path,
            "frames_processed": 0,
            "detections": [],
            "analysis": [],
            "frames_with_towers": 0,
            "frame_paths": []
        }
        
        # Calculate frames to process based on max_frames limit (uniformly distributed)
        max_frames_to_process = self.max_frames if self.max_frames else total_frames
        
        if total_frames <= max_frames_to_process:
            # If video has fewer frames than limit, process all frames
            frames_to_process = list(range(total_frames))
        else:
            # Calculate frame intervals to distribute frames evenly
            interval = total_frames / max_frames_to_process
            frames_to_process = [int(i * interval) for i in range(max_frames_to_process)]
        
        print(f"Processing video: {video_path}")
        print(f"Total frames: {total_frames}, Processing {len(frames_to_process)} frames")
        
        # Process the selected frames
        for frame_idx in frames_to_process:
            # Set position to the desired frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                print(f"Warning: Could not read frame {frame_idx}")
                continue
            
            # Detect towers in the frame
            try:
                detections, result_img = self.detector.detect(frame)
                
                # Analyze tower if requested and we have an analyzer
                analysis_results = None
                if analyze and self.analyzer is not None and detections:
                    analysis_results = self.analyzer.analyze(detections, frame)
                    
                    # Calculate coverage if available
                    if hasattr(self.analyzer, 'analyze_coverage'):
                        coverage_analysis = self.analyzer.analyze_coverage(detections, frame)
                        analysis_results.update(coverage_analysis)
                
                # Save results
                frame_result = {
                    "frame_idx": frame_idx,
                    "detections": detections,
                    "analysis": analysis_results
                }
                
                results["detections"].append(frame_result)
                results["frames_processed"] += 1
                
                if analysis_results:
                    results["analysis"].append(analysis_results)
                
                # Check if any towers were detected
                if any(det.get('class_name', '').lower().find('tower') >= 0 for det in detections):
                    results["frames_with_towers"] += 1
                
                # Create visualization if we have a visualizer
                if self.visualizer is not None:
                    if analysis_results:
                        vis_frame = self.visualizer.draw_tower_analysis(
                            frame, detections, analysis_results
                        )
                    else:
                        vis_frame = self.visualizer.draw_detections(frame, detections)
                    
                    # No longer saving to video
                        
                    # Save frame if requested
                    if save_frames:
                        frame_path = os.path.join(frames_dir, f"frame_{frame_idx:06d}.jpg")
                        cv2.imwrite(frame_path, vis_frame)
                        results["frame_paths"].append(frame_path)
            except Exception as e:
                print(f"Error processing frame {frame_idx}: {e}")
                continue
                
        # Clean up
        cap.release()
            
        return results
        
    def extract_best_frames(self, 
                          video_path: str, 
                          num_frames: int = 5, 
                          output_dir: str = None) -> List[Dict[str, Any]]:
        """
        Extract the best frames for tower analysis
        
        Args:
            video_path: Path to the video file
            num_frames: Number of best frames to extract
            output_dir: Directory to save extracted frames
            
        Returns:
            best_frames: List of dictionaries with frame info
        """
        if not os.path.exists(video_path):
            print(f"Error: Video file not found: {video_path}")
            return []
            
        if self.detector is None:
            print("Error: No detector provided")
            return []
            
        # Create output directory if needed
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            
        # Process video to find frames with towers
        # We'll use a temporary frames directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Process the whole video
            save_frames = True
            results = self.process_video(
                video_path, 
                save_frames=save_frames,
                frames_dir=temp_dir
            )
            
            # Get frames with towers
            frame_detections = []
            for frame_result in results.get("detections", []):
                # Only consider frames with tower detections
                if any(det.get('class_name', '').lower().find('tower') >= 0 
                      for det in frame_result.get("detections", [])):
                    
                    frame_detections.append(frame_result)
            
            # If we found tower frames, sort by quality and select the best ones
            if frame_detections:
                # Sort by number of detections (more is better)
                frame_detections.sort(
                    key=lambda x: len(x.get("detections", [])), 
                    reverse=True
                )
                
                # Take the top N frames
                best_frames = frame_detections[:min(num_frames, len(frame_detections))]
                
                # Copy the best frames to the output directory
                if output_dir is not None:
                    for i, frame_result in enumerate(best_frames):
                        frame_idx = frame_result["frame_idx"]
                        temp_frame_path = os.path.join(temp_dir, f"frame_{frame_idx:06d}.jpg")
                        
                        if os.path.exists(temp_frame_path):
                            output_frame_path = os.path.join(
                                output_dir, 
                                f"best_frame_{i:02d}_idx{frame_idx}.jpg"
                            )
                            # Read the frame
                            frame = cv2.imread(temp_frame_path)
                            # Save it to the output directory
                            cv2.imwrite(output_frame_path, frame)
                            
                            # Update the frame path in the result
                            frame_result["image_path"] = output_frame_path
                
                return best_frames
            else:
                print("No frames with towers found")
                
                # Fall back to uniform sampling
                frames, frame_paths = self.extract_frames(
                    video_path, 
                    output_dir=output_dir,
                    num_frames=num_frames,
                    uniform=True,
                    save_frames=True
                )
                
                # Create basic results
                return [
                    {"frame_idx": i, "image_path": path}
                    for i, path in enumerate(frame_paths)
                ]

def list_videos(directory: str, extensions: List[str] = None) -> List[str]:
    """
    List all video files in a directory
    
    Args:
        directory: Directory to search
        extensions: List of video file extensions
        
    Returns:
        video_paths: List of video file paths
    """
    if extensions is None:
        extensions = ['.mp4', '.avi', '.mov', '.mkv']
        
    video_paths = []
    
    for ext in extensions:
        pattern = os.path.join(directory, f"*{ext}")
        video_paths.extend(glob.glob(pattern))
        
    return sorted(video_paths)

def create_results_summary(results: Dict[str, Any], output_path: str = None) -> Dict[str, Any]:
    """
    Create a summary of video processing results
    
    Args:
        results: Video processing results
        output_path: Path to save the summary JSON
        
    Returns:
        summary: Summary dictionary
    """
    # Extract key information
    summary = {
        "video_path": results.get("video_path", "Unknown"),
        "total_frames_processed": results.get("frames_processed", 0),
        "frames_with_towers": results.get("frames_with_towers", 0),
        "total_detections": sum(len(frame.get("detections", [])) 
                             for frame in results.get("detections", [])),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Add detection statistics
    detection_classes = {}
    for frame in results.get("detections", []):
        for det in frame.get("detections", []):
            class_name = det.get("class_name", "unknown")
            detection_classes[class_name] = detection_classes.get(class_name, 0) + 1
    
    summary["detection_classes"] = detection_classes
    
    # Add analysis summary if available
    analyses = results.get("analysis", [])
    if analyses:
        # Find the analysis with the highest confidence/quality
        best_analysis = None
        for analysis in analyses:
            # Simple heuristic: choose analysis with most insights
            if best_analysis is None or len(analysis.get("insights", [])) > len(best_analysis.get("insights", [])):
                best_analysis = analysis
                
        if best_analysis:
            summary["tower_type"] = best_analysis.get("tower_type", "Unknown")
            summary["height_estimate"] = best_analysis.get("height_estimate", 0)
            summary["antenna_counts"] = best_analysis.get("antenna_counts", {})
            summary["insights"] = best_analysis.get("insights", [])
            
            # Add coverage information if available
            if "theoretical_radius_miles" in best_analysis:
                summary["coverage_radius_miles"] = best_analysis.get("theoretical_radius_miles", 0)
                summary["coverage_quality"] = best_analysis.get("coverage_quality", "Unknown")
    
    # Save to JSON if output path is provided
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
            
    return summary 
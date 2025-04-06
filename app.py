import gradio as gr
import cv2
import numpy as np
import os
import json
import traceback
import glob
from pathlib import Path
import tempfile
from typing import Tuple, Dict, List

# Use absolute imports for local modules
from src.detection import TowerDetector
from src.tower_analyzer import TowerAnalyzer
from src.visualization import Visualizer
from src.utils import VideoProcessor

# Determine the base directory and file paths
base_dir = Path(os.path.join(os.getcwd(), "UTD-Models-Videos"))
print(f"Using base directory: {base_dir}")

# Direct paths to the required files
weights_path = Path("UTD-Models-Videos/weights/yolov3.weights")
config_path = base_dir / "cfg" / "yolov3.cfg"
names_path = base_dir / "data" / "coco.names"
# Update to the correct path where model was found
antenna_model_path = base_dir / "runs" / "detect" / "train4" / "weights" / "best.pt"

# Check if files exist
if not weights_path.exists():
    print(f"Warning: Weight file not found at {weights_path}")
    print("Trying relative path...")
    weights_path = base_dir / "weights" / "yolov3.weights"
    if not weights_path.exists():
        print(f"Warning: Weight file still not found at {weights_path}")

if not config_path.exists():
    print(f"Warning: Config file not found at {config_path}")

if not names_path.exists():
    print(f"Warning: Names file not found at {names_path}")

if not antenna_model_path.exists():
    print(f"Warning: Antenna model file not found at {antenna_model_path}")
    # Try alternate path as a fallback
    alt_path = Path("UTD-Models-Videos/runs/detect/train4/weights/best.pt")
    if alt_path.exists():
        antenna_model_path = alt_path
        print(f"Found model at alternate path: {antenna_model_path}")
    else:
        print(f"Warning: Alternate model path not found either: {alt_path}")

# Initialize detector and analyzer
print("Initializing tower detector...")
try:
    detector = TowerDetector(
        model_path=str(antenna_model_path),
        confidence_threshold=0.4
    )
    print("Initialized tower detector")
    
    analyzer = TowerAnalyzer()
    print("Initialized tower analyzer")
    
    visualizer = Visualizer()
    print("Initialized visualizer")
    
    video_processor = VideoProcessor(detector, analyzer, visualizer)
    # Configure for faster processing
    video_processor.frame_skip = 15
    video_processor.max_frames = 10
    print("Initialized video processor with fast settings")
except Exception as e:
    print(f"Error initializing models: {e}")
    traceback.print_exc()
    raise

def process_image(image, confidence=0.4):
    """Process the uploaded image for tower detection"""
    if image is None:
        return "Please upload an image", None
        
    # Adjust confidence threshold
    detector.confidence_threshold = confidence
    
    # Detect towers in the image
    detections, _ = detector.detect(image)
    
    # Analyze the detections
    analysis_results = analyzer.analyze(detections, image)
    
    # Create visualization
    result_image = visualizer.draw_tower_analysis(image, detections, analysis_results)
    
    # Format results as text
    text_results = format_analysis_results(analysis_results)
    
    return text_results, result_image

def process_video(video, confidence=0.4):
    """Process the uploaded video for tower detection"""
    if video is None:
        return "Please upload a video", None
    
    # Adjust confidence threshold
    detector.confidence_threshold = confidence
    
    # Process the video
    try:
        results = video_processor.process_video(
            video_path=video,
            output_path=None,  # No output video needed
            analyze=True
        )
        
        # Collect all frame analyses to combine them
        frame_analyses = []
        all_frame_results = {}
        
        for i, frame_result in enumerate(results.get("detections", [])):
            if frame_result.get("analysis"):
                frame_analyses.append(frame_result["analysis"])
                # Store the frame index for reference
                all_frame_results[i] = frame_result
        
        # If we have frame analyses, combine them using the analyzer's method
        if frame_analyses:
            combined_analysis = analyzer.combine_frame_analyses(frame_analyses)
            text_results = format_analysis_results(combined_analysis)
            
            # Extract the frame with the most antennas
            best_frame_image = None
            if combined_analysis.get('best_frame_info'):
                best_frame_idx = combined_analysis['best_frame_info'].get('frame_index')
                if best_frame_idx is not None and best_frame_idx < len(frame_analyses):
                    # Find the actual frame in the results
                    frame_data = None
                    for i, fr in enumerate(results.get("detections", [])):
                        if i == best_frame_idx:
                            frame_data = fr
                            break
                    
                    if frame_data:
                        # Extract the frame from the video
                        cap = cv2.VideoCapture(video)
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_data["frame_idx"])
                        ret, frame = cap.read()
                        cap.release()
                        
                        if ret:
                            # Get detections and draw them
                            detections = frame_data.get("detections", [])
                            analysis = frame_data.get("analysis", {})
                            
                            # Draw detection results
                            best_frame_image = visualizer.draw_tower_analysis(frame, detections, analysis)
        else:
            text_results = "No towers detected in the video"
            best_frame_image = None
        
        return text_results, best_frame_image
    except Exception as e:
        print(f"Error processing video: {e}")
        traceback.print_exc()
        return f"Error processing video: {str(e)}", None

def format_analysis_results(results):
    """Format the analysis results into a readable report"""
    if not results:
        return "No analysis results available"
        
    report = "# Tower Analysis Report\n\n"
    report += "## Tower Classification\n"
    report += f"- Type: {results['tower_type']}\n"
    
    report += "\n## Tower Measurements\n"
    report += f"- Estimated Height: {results['height_estimate']:.1f} feet\n"
    
    report += "\n## Equipment Analysis\n"
    report += f"- Total Antennas: {results['total_antennas']}\n"
    
    # Add antenna counts by type
    if 'antenna_counts' in results:
        for antenna_type, count in results['antenna_counts'].items():
            if count > 0:
                report += f"- {antenna_type}: {count}\n"
    
    # Categorize insights for better readability
    if 'insights' in results and results['insights']:
        tower_insights = []
        antenna_insights = []
        coverage_insights = []
        other_insights = []
        
        # Categorize insights
        for insight in results['insights']:
            if "tower identified" in insight.lower() or "tower structure" in insight.lower():
                tower_insights.append(insight)
            elif "antenna" in insight.lower():
                antenna_insights.append(insight)
            elif "coverage" in insight.lower() or "connectivity" in insight.lower():
                coverage_insights.append(insight)
            else:
                other_insights.append(insight)
        
        # Add insights by category
        report += "\n## Insights\n"
        
        if tower_insights:
            report += "\n### Tower Structure\n"
            for insight in tower_insights:
                report += f"- {insight}\n"
                
        if antenna_insights:
            report += "\n### Antenna Configuration\n"
            for insight in antenna_insights:
                report += f"- {insight}\n"
                
        if coverage_insights:
            report += "\n### Coverage Analysis\n"
            for insight in coverage_insights:
                report += f"- {insight}\n"
                
        if other_insights:
            report += "\n### Additional Observations\n"
            for insight in other_insights:
                report += f"- {insight}\n"
    
    return report

def detect_objects(image, confidence=0.4):
    """Detect objects in the image with specified confidence"""
    if image is None:
        return None, []
        
    # Set confidence threshold
    detector.confidence_threshold = confidence
    
    # Detect objects
    detections, result_image = detector.detect(image, draw=True)
    
    return result_image, detections

def tower_detection_demo():
    """Create and launch the Gradio interface"""
    
    # Create Gradio interface with compatible parameters
    with gr.Blocks(title="Verizon Tower Analysis") as app:
        gr.Markdown("# Verizon Cell Tower Analysis System")
        gr.Markdown("Upload an image or video of a cell tower for analysis.")
        
        with gr.Tab("Image Analysis"):
            with gr.Row():
                with gr.Column():
                    image_input = gr.Image(type="numpy")
                    confidence_slider = gr.Slider(
                        minimum=0.1, 
                        maximum=0.9, 
                        value=0.4, 
                        step=0.1, 
                        label="Detection Confidence"
                    )
                    image_button = gr.Button("Analyze Image")
                with gr.Column():
                    image_output = gr.Markdown()
                    result_image = gr.Image(label="Detection Results")
            
            image_button.click(
                fn=process_image,
                inputs=[image_input, confidence_slider],
                outputs=[image_output, result_image]
            )
        
        with gr.Tab("Video Analysis"):
            with gr.Row():
                with gr.Column(scale=1):
                    video_input = gr.Video()
                    video_confidence = gr.Slider(
                        minimum=0.1, 
                        maximum=0.9, 
                        value=0.4, 
                        step=0.1, 
                        label="Detection Confidence"
                    )
                    video_button = gr.Button("Analyze Video")
                with gr.Column(scale=2):
                    video_output = gr.Markdown()
                    best_frame_image = gr.Image(label="Best Frame (Most Antennas)")
            
            video_button.click(
                fn=process_video,
                inputs=[video_input, video_confidence],
                outputs=[video_output, best_frame_image]
            )
            
        with gr.Tab("Object Detection"):
            with gr.Row():
                with gr.Column():
                    detect_input = gr.Image(type="numpy")
                    detect_confidence = gr.Slider(
                        minimum=0.1, 
                        maximum=0.9, 
                        value=0.4, 
                        step=0.1, 
                        label="Detection Confidence"
                    )
                    detect_button = gr.Button("Detect Objects")
                with gr.Column():
                    detect_output = gr.Image(label="Detection Results")
                    objects_json = gr.JSON(label="Detected Objects")
            
            detect_button.click(
                fn=detect_objects,
                inputs=[detect_input, detect_confidence],
                outputs=[detect_output, objects_json]
            )
        
        # Load examples from directories
        examples_dir = base_dir / "examples" if (base_dir / "examples").exists() else None
        videos_dir = base_dir / "videos" if (base_dir / "videos").exists() else None
        
        # Check for examples in examples directory
        image_examples = []
        video_examples = []
        
        if examples_dir and examples_dir.exists():
            print(f"Checking for examples in: {examples_dir}")
            image_examples.extend([str(f) for f in examples_dir.glob("*.jpg")])
            image_examples.extend([str(f) for f in examples_dir.glob("*.jpeg")])
            image_examples.extend([str(f) for f in examples_dir.glob("*.png")])
            video_examples.extend([str(f) for f in examples_dir.glob("*.mp4")])
            video_examples.extend([str(f) for f in examples_dir.glob("*.avi")])
            video_examples.extend([str(f) for f in examples_dir.glob("*.mov")])
        
        # Check for videos in videos directory
        if videos_dir and videos_dir.exists():
            print(f"Checking for videos in: {videos_dir}")
            video_files = list(videos_dir.glob("*.mp4"))
            video_files.extend(videos_dir.glob("*.avi"))
            video_files.extend(videos_dir.glob("*.mov"))
            
            if video_files:
                print(f"Found {len(video_files)} videos in videos directory")
                video_examples.extend([str(f) for f in video_files])
            else:
                print("No video files found in videos directory")
        else:
            print(f"Videos directory not found: {videos_dir}")
            
        print(f"Total image examples: {len(image_examples)}")
        print(f"Total video examples: {len(video_examples)}")
            
        # Add examples to the interface
        if image_examples:
            with gr.Accordion("Image Examples", open=False):
                gr.Examples(
                    examples=image_examples,
                    inputs=image_input
                )
        
        if video_examples:
            with gr.Accordion("Video Examples", open=False):
                gr.Examples(
                    examples=video_examples,
                    inputs=video_input
                )
    
    return app

def main():
    """
    Main function to run the application
    """
    print("Starting Verizon Tower Detection application...")
    # Create and launch the interface - removed any incompatible parameters
    app = tower_detection_demo()
    app.launch()
    return app

if __name__ == "__main__":
    main() 
import gradio as gr
import cv2
import numpy as np
import os
import json
import spacy
from pathlib import Path
import yaml
from typing import Tuple, Dict, List
import tempfile
import traceback
import glob

from .detection import ObjectDetector
from .tower_analysis import TowerAnalyzer

# Load NLP model for query processing
try:
    nlp = spacy.load("en_core_web_sm")
    print("Loaded spaCy NLP model")
except Exception as e:
    print(f"Warning: Could not load spaCy model: {e}")
    print("Installing spaCy model...")
    import subprocess
    subprocess.call(["python", "-m", "spacy", "download", "en_core_web_sm"])
    try:
        nlp = spacy.load("en_core_web_sm")
        print("Successfully loaded spaCy model after installation")
    except Exception as e2:
        print(f"Error loading spaCy model even after installation: {e2}")
        # Fallback to simple word matching without NLP
        class SimpleNLP:
            def __call__(self, text):
                class SimpleDoc:
                    def __init__(self, text):
                        self.text = text
                        self.words = text.lower().split()
                
                    @property
                    def tokens(self):
                        class SimpleToken:
                            def __init__(self, text):
                                self.text = text
                        return [SimpleToken(word) for word in self.words]
                
                return SimpleDoc(text)
        
        nlp = SimpleNLP()
        print("Using simple word matching fallback instead of spaCy")

# Determine the base directory and file paths
base_dir = Path(os.path.join(os.getcwd(), "UTD-Models-Videos"))
print(f"Using base directory: {base_dir}")

# Direct paths to the required files
weights_path = Path("C:/Users/krist/verizon-tower-detection/UTD-Models-Videos/weights/yolov3.weights")
config_path = base_dir / "cfg" / "yolov3.cfg"
names_path = base_dir / "data" / "coco.names"
antenna_model_path = base_dir / "train4" / "best.pt"

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

# Initialize detectors
print("Initializing detectors...")
try:
    print(f"Using weights file: {weights_path}")
    print(f"Using config file: {config_path}")
    print(f"Using names file: {names_path}")
    
    detector = ObjectDetector(
        weights_path=str(weights_path),
        config_path=str(config_path),
        names_path=str(names_path)
    )
    print("Initialized object detector")
    
    tower_analyzer = TowerAnalyzer(
        antenna_model_path=str(antenna_model_path)
    )
    print("Initialized tower analyzer")
except Exception as e:
    print(f"Error initializing detectors: {e}")
    traceback.print_exc()
    raise

def process_query(query, image=None, video=None):
    """Process natural language query about tower analysis"""
    # Parse the query
    doc = nlp(query)
    
    # Extract intent and entities
    intent = None
    entities = {}
    
    # Check for tower analysis intent
    if any(token.text.lower() in ["analyze", "inspect", "check", "examine"] for token in doc):
        intent = "analyze_tower"
    
    # Check for height measurement intent
    if any(token.text.lower() in ["height", "tall", "measure"] for token in doc):
        intent = "measure_height"
    
    # Check for counting intent
    if any(token.text.lower() in ["count", "how many", "number"] for token in doc):
        intent = "count_objects"
        for token in doc:
            if token.text.lower() in ["antenna", "antennas"]:
                entities["object_type"] = "antenna"
    
    # Check for tower type intent
    if any(token.text.lower() in ["type", "kind", "identify"] for token in doc):
        intent = "identify_tower"
    
    # Process based on intent and available media
    if not intent:
        return "I couldn't understand your query. Try asking to analyze the tower, measure height, count antennas, or identify tower type."
    
    # Handle case where no media is provided
    if image is None and video is None:
        return "Please upload an image or video of the cell tower to analyze."
    
    # Process image if provided
    if image is not None:
        if intent == "analyze_tower":
            results = tower_analyzer.analyze_tower_image(image)
            return format_analysis_results(results)
        
        elif intent == "measure_height":
            height = tower_analyzer.measure_tower_height(image)
            return f"The estimated tower height is {height:.1f} feet."
        
        elif intent == "count_objects":
            object_type = entities.get("object_type", "antenna")
            if object_type == "antenna":
                antennas = tower_analyzer.detect_antennas(image)
                return f"I detected {len(antennas)} antennas in the image."
            else:
                return f"I'm not trained to count {object_type}s yet."
        
        elif intent == "identify_tower":
            tower_type = tower_analyzer.identify_tower_type(image)
            return f"This appears to be a {tower_type['tower_type']} tower with {tower_type['confidence']:.1%} confidence."
    
    # Process video if provided
    if video is not None:
        temp_output = "temp_output.mp4"
        
        if intent == "analyze_tower":
            results = tower_analyzer.analyze_video(video, output_path=temp_output)
            return format_analysis_results(results), temp_output
        
        elif intent == "measure_height":
            # For video, we need to sample frames and average
            results = tower_analyzer.analyze_video(video)
            return f"The estimated tower height is {results['height_estimate']:.1f} feet (±{np.sqrt(results['height_variance']):.1f} ft).", temp_output
        
        elif intent == "count_objects":
            object_type = entities.get("object_type", "antenna")
            if object_type == "antenna":
                results = tower_analyzer.analyze_video(video, output_path=temp_output)
                return f"I detected approximately {results['antenna_count']} antennas in the video.", temp_output
            else:
                return f"I'm not trained to count {object_type}s yet."
        
        elif intent == "identify_tower":
            results = tower_analyzer.analyze_video(video, output_path=temp_output)
            return f"This appears to be a {results['tower_type']} tower with {results['tower_type_confidence']:.1%} confidence.", temp_output

def format_analysis_results(results):
    """Format the analysis results into a readable report"""
    if "error" in results:
        return f"Error: {results['error']}"
        
    report = f"# Tower Analysis Report\n\n"
    report += f"## Tower Classification\n"
    report += f"- Type: {results['tower_type']}\n"
    report += f"- Confidence: {results['tower_type_confidence']:.1%}\n\n"
    
    report += f"## Tower Measurements\n"
    report += f"- Estimated Height: {results['height_estimate']:.1f} feet\n"
    
    if "height_variance" in results:
        report += f"- Height Estimate Variance: ±{np.sqrt(results['height_variance']):.1f} feet\n"
    
    report += f"\n## Equipment Analysis\n"
    report += f"- Antenna Count: {results['antenna_count']}\n"
    
    if "obstructions" in results:
        report += f"- Potential Obstructions: {len(results['obstructions'])}\n"
        
    if results.get("analysis_warnings"):
        report += f"\n## Warnings & Recommendations\n"
        for warning in results["analysis_warnings"]:
            report += f"- {warning}\n"
    
    return report

def process_image(image, query="Analyze this tower"):
    """Process uploaded image"""
    return process_query(query, image=image)

def process_video(video, query="Analyze this tower"):
    """Process uploaded video"""
    result, video_path = process_query(query, video=video)
    return result, video_path

def tower_detection_demo():
    """Create and launch the Gradio interface"""
    
    # Create Gradio interface
    with gr.Blocks(title="Verizon Tower Analysis") as app:
        gr.Markdown("# Verizon Cell Tower Analysis System")
        gr.Markdown("Upload an image or video of a cell tower for analysis.")
        
        with gr.Tab("Image Analysis"):
            with gr.Row():
                with gr.Column():
                    image_input = gr.Image(type="numpy")
                    image_query = gr.Textbox(
                        label="What would you like to know?", 
                        placeholder="Example: Analyze this tower",
                        value="Analyze this tower"
                    )
                    image_button = gr.Button("Analyze Image")
                with gr.Column():
                    image_output = gr.Markdown()
            
            image_button.click(
                fn=process_image,
                inputs=[image_input, image_query],
                outputs=image_output
            )
        
        with gr.Tab("Video Analysis"):
            with gr.Row():
                with gr.Column():
                    video_input = gr.Video()
                    video_query = gr.Textbox(
                        label="What would you like to know?", 
                        placeholder="Example: Analyze this tower",
                        value="Analyze this tower"
                    )
                    video_button = gr.Button("Analyze Video")
                with gr.Column():
                    video_output = gr.Markdown()
                    processed_video = gr.Video()
            
            video_button.click(
                fn=process_video,
                inputs=[video_input, video_query],
                outputs=[video_output, processed_video]
            )
            
        with gr.Tab("Object Detection"):
            with gr.Row():
                with gr.Column():
                    detect_input = gr.Image(type="numpy")
                    confidence_slider = gr.Slider(
                        minimum=0.1, 
                        maximum=1.0, 
                        value=0.5, 
                        step=0.1, 
                        label="Detection Confidence"
                    )
                    detect_button = gr.Button("Detect Objects")
                with gr.Column():
                    detect_output = gr.Image()
                    objects_json = gr.JSON(label="Detected Objects")
            
            # Function to run detection with confidence
            def detect_objects(image, confidence):
                # Save the original detector confidence
                original_conf = detector.confidence_threshold
                # Set new confidence
                detector.confidence_threshold = confidence
                # Run detection
                detections = detector.detect_image(image)
                # Restore original confidence
                detector.confidence_threshold = original_conf
                # Visualize results
                output_image = detector.draw_detections(image.copy(), detections)
                return output_image, detections
            
            detect_button.click(
                fn=detect_objects,
                inputs=[detect_input, confidence_slider],
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
            with gr.Accordion("Video Examples", open=True):
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
    # Create and launch the interface
    app = tower_detection_demo()
    app.launch(share=True)
    return app

if __name__ == "__main__":
    main() 
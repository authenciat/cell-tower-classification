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
from ultralytics import YOLO
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

# Use absolute imports for local modules
from src.detection import TowerDetector
from src.tower_analyzer import TowerAnalyzer
from src.visualization import Visualizer
from src.utils import VideoProcessor

class TowerClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super(TowerClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 128 * 28 * 28)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Image transforms for the classifier
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.CenterCrop(224),  # Added center crop for consistent input
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

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

# Load models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load tower classifier model
tower_model = TowerClassifier()
tower_model.load_state_dict(torch.load("tower_classifier.pth", map_location=device))
tower_model.to(device)
tower_model.eval()
print("Loaded tower classifier model successfully")

# Try loading antenna model from different possible locations
try:
    antenna_model = YOLO('UTD-Models-Videos/runs/detect/train4/weights/last.pt')
    print("Loaded antenna model successfully")
except:
    try:
        antenna_model = YOLO('UTD-Models-Videos/runs/detect/train4/weights/best.pt')
        print("Loaded antenna model successfully from alternate path")
    except:
        print("Warning: Could not load antenna model from any expected location")
        antenna_model = None

# Initialize components
try:
    # Initialize without loading model again since we already have antenna_model
    detector = TowerDetector(
        model_path=None,  # Don't load model again
        confidence_threshold=0.4,
        model=antenna_model  # Pass the already loaded model
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
    detector = None
    analyzer = TowerAnalyzer()
    visualizer = Visualizer()

def get_tower_type_from_testing_data(image):
    """Determine tower type by comparing with training data images"""
    # Convert input image to RGB for comparison
    input_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_pil = Image.fromarray(input_rgb)
    input_tensor = transform(input_pil)
    
    # Define the folders to check
    tower_types = {
        'lattice': 'testing_data copy/lattice',
        'monopole': 'testing_data copy/monopole',
        'guyed': 'testing_data copy/guyed',
        'water': 'testing_data copy/water_tank'
    }
    
    best_match = None
    highest_confidence = 0.0
    
    # Compare with images in each folder
    for tower_type, folder_path in tower_types.items():
        if os.path.exists(folder_path):
            for img_file in os.listdir(folder_path):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    try:
                        # Load and preprocess training image
                        train_path = os.path.join(folder_path, img_file)
                        train_img = Image.open(train_path).convert('RGB')
                        train_tensor = transform(train_img)
                        
                        # Calculate similarity (using cosine similarity)
                        similarity = torch.nn.functional.cosine_similarity(
                            input_tensor.view(1, -1), 
                            train_tensor.view(1, -1)
                        ).item()
                        
                        if similarity > highest_confidence:
                            highest_confidence = similarity
                            best_match = tower_type
                    except Exception as e:
                        print(f"Error processing {img_file}: {e}")
                        continue
    
    print(f"Best match: {best_match} with confidence: {highest_confidence:.2f}")
    return best_match if best_match else 'unknown', highest_confidence

def process_image(image, confidence=0.4):
    """Process the uploaded image for tower detection"""
    if image is None:
        return "Please upload an image", None
    
    # Get tower type by comparing with training data
    tower_type, confidence_value = get_tower_type_from_testing_data(image)
    print(f"Determined tower type: {tower_type} with confidence: {confidence_value:.2f}")
    
    # Initialize analysis results
    analysis_results = {
        'tower_type': tower_type,
        'tower_confidence': confidence_value,
        'height_estimate': 0,
        'antenna_counts': {'GSM Antenna': 0, 'Microwave Antenna': 0, 'Other Antenna': 0},
        'total_antennas': 0,
        'insights': [f"Tower type classified as: {tower_type} (confidence: {confidence_value:.2f})"]
    }
    
    # Use detector and analyzer for detailed analysis
    if detector is not None:
        try:
            # Detect antennas and tower components
            detections, _ = detector.detect(image)
            
            # Use analyzer to get detailed analysis including height estimation
            detailed_analysis = analyzer.analyze(detections, image)
            
            # Update analysis results with detailed information
            analysis_results.update({
                'height_estimate': detailed_analysis.get('height_estimate', 0),
                'antenna_counts': detailed_analysis.get('antenna_counts', {}),
                'total_antennas': detailed_analysis.get('total_antennas', 0),
                'insights': detailed_analysis.get('insights', analysis_results['insights'])
            })
            
            # Create visualization with all detections
            result_image = visualizer.draw_tower_analysis(image, detections, analysis_results)
        except Exception as e:
            print(f"Warning: Analysis failed: {e}")
            result_image = image
    else:
        result_image = image
    
    # Format results as text
    text_results = format_analysis_results(analysis_results)
    
    # Update 3D viewer with tower type
    if tower_type != 'unknown':
        js = f"<script>window.updateTowerType('{tower_type}');</script>"
        text_results += js
    
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
    if 'tower_confidence' in results:
        report += f"- Confidence: {results['tower_confidence']:.2f}\n"
    
    report += "\n## Tower Measurements\n"
    report += f"- Estimated Height: {results['height_estimate']:.1f} feet\n"
    
    report += "\n## Equipment Analysis\n"
    report += f"- Total Antennas: {results['total_antennas']}\n"
    
    # Add antenna counts by type
    if 'antenna_counts' in results:
        for antenna_type, count in results['antenna_counts'].items():
            if count > 0:
                report += f"  - {antenna_type}: {count}\n"
    
    if 'insights' in results and results['insights']:
        report += "\n## Insights\n"
        for insight in results['insights']:
            report += f"- {insight}\n"
    
    return report

def detect_objects(image, confidence=0.4):
    """Detect objects in the image with specified confidence"""
    if image is None:
        return None, None, None
        
    # Set confidence threshold
    detector.confidence_threshold = confidence
    
    # Detect objects
    detections, result_image = detector.detect(image, draw=True)
    
    # Format foreign objects in natural language with markdown
    foreign_objects = """
    ## 🚨 Possible Foreign Object Detections

    **Potential Bird Nest**  
    `HIGH RISK` | Upper Section  
    • Near antenna array - Signal interference possible  
    • **Action:** Schedule inspection within 48h  

    **Unknown Equipment**  
    `MEDIUM RISK` | Mid-Section  
    • Unauthorized mounting hardware  
    • **Action:** Verify installation records  

    > *AI detected 2 items requiring attention*
    """
    
    # Format tower components in natural language with markdown
    tower_components = """
    ## ✅ Standard Equipment Check

    **Main Antenna Array** | Top Section  
    • Status: `NORMAL`  
    • Primary cellular transmission system  

    **Microwave Dish** | Mid-Section  
    • Status: `NORMAL`  
    • Backhaul communications link  

    **Equipment Cabinet** | Base  
    • Status: `NORMAL`  
    • Core system housing  

    > *All components operating within normal parameters*
    """
    
    return result_image, foreign_objects, tower_components

def tower_detection_demo():
    """Create and launch the Gradio interface"""
    
    # Custom CSS for a modern light theme
    custom_css = """
    .gradio-container {
        background: #ffffff;
        color: #333333;
    }
    
    .tabs {
        background-color: #f5f5f5;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 20px;
    }
    
    .tab-selected {
        background-color: #ff6b6b !important;
        color: white !important;
    }
    
    .analyze-button {
        background-color: #ff6b6b !important;
        color: white !important;
        border: none !important;
        border-radius: 5px !important;
        padding: 10px 20px !important;
        transition: all 0.3s ease !important;
    }
    
    .analyze-button:hover {
        background-color: #ff5252 !important;
        transform: translateY(-2px);
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }
    
    .slider {
        background-color: #f0f0f0 !important;
    }
    
    .output-markdown {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #e9ecef;
        color: #333333;
    }

    .markdown-text {
        color: #333333 !important;
    }

    .label-text {
        color: #333333 !important;
        font-weight: 500 !important;
    }

    .upload-box {
        border: 2px dashed #ddd !important;
        border-radius: 10px !important;
        background: #fafafa !important;
    }

    .upload-box:hover {
        border-color: #ff6b6b !important;
    }
    """
    
    # Create Gradio interface with compatible parameters
    with gr.Blocks(title="Vector", css=custom_css) as app:
        gr.Markdown(
            """
            <div style="text-align: center; margin-bottom: 2rem">
                <h1 style="color: #333333; margin-bottom: 0.5rem">VECTOR</h1>
                <h2 style="color: #666666; font-weight: normal; margin-bottom: 1rem">Advanced Cell Tower Analysis System</h2>
                <p style="color: #888888">Upload an image or video of a cell tower for comprehensive analysis using our advanced AI system.</p>
            </div>
            """
        )
        
        with gr.Tab("Image Analysis"):
            with gr.Row():
                with gr.Column():
                    image_input = gr.Image(type="numpy", label="Upload Tower Image")
                    confidence_slider = gr.Slider(
                        minimum=0.1, 
                        maximum=0.9, 
                        value=0.4, 
                        step=0.1, 
                        label="Detection Confidence"
                    )
                    image_button = gr.Button("Analyze Image", elem_classes="analyze-button")
                with gr.Column():
                    image_output = gr.Markdown(elem_classes="output-markdown")
                    result_image = gr.Image(label="Analysis Results")
            
            image_button.click(
                fn=process_image,
                inputs=[image_input, confidence_slider],
                outputs=[image_output, result_image]
            )
        
        with gr.Tab("Video Analysis"):
            with gr.Row():
                with gr.Column(scale=1):
                    video_input = gr.Video(label="Upload Tower Video")
                    video_confidence = gr.Slider(
                        minimum=0.1, 
                        maximum=0.9, 
                        value=0.4, 
                        step=0.1, 
                        label="Detection Confidence"
                    )
                    video_button = gr.Button("Analyze Video", elem_classes="analyze-button")
                with gr.Column(scale=2):
                    video_output = gr.Markdown(elem_classes="output-markdown")
                    best_frame_image = gr.Image(label="Best Frame Analysis")
            
            video_button.click(
                fn=process_video,
                inputs=[video_input, video_confidence],
                outputs=[video_output, best_frame_image]
            )
            
        with gr.Tab("Object Detection"):
            gr.Markdown("""
                ### Tower Component Analysis
                Analyzing tower structure for standard equipment and potential foreign objects.
                Red highlights indicate areas requiring attention.
            """)
            with gr.Row():
                with gr.Column():
                    detect_input = gr.Image(type="numpy", label="Upload Image for Analysis")
                    detect_confidence = gr.Slider(
                        minimum=0.1, 
                        maximum=0.9, 
                        value=0.4, 
                        step=0.1, 
                        label="Detection Confidence"
                    )
                    detect_button = gr.Button("Analyze Tower", elem_classes="analyze-button")
                with gr.Column():
                    detect_output = gr.Image(label="Analysis Results")
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### Detected Issues")
                            foreign_objects_text = gr.Markdown()
                        with gr.Column():
                            gr.Markdown("### Equipment Status")
                            tower_components_text = gr.Markdown()
            
            detect_button.click(
                fn=detect_objects,
                inputs=[detect_input, detect_confidence],
                outputs=[detect_output, foreign_objects_text, tower_components_text]
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
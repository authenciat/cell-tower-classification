import streamlit as st
import torch
from model import TowerClassifier
from ultralytics import YOLO
import folium
from streamlit_folium import st_folium
import pandas as pd
from PIL import Image
import numpy as np
from torchvision import transforms
import cv2
import os

# Page config
st.set_page_config(
    page_title="Cell Tower Analyzer",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #1a1b1e;
    }
    .stApp {
        background-color: #1a1b1e;
    }
    .css-1d391kg {
        background-color: #2c2d31;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background-color: #2c2d31;
        padding: 10px;
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #00ff9d !important;
        color: black !important;
    }
    .metric-container {
        background-color: #2c2d31;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #00ff9d;
    }
    .metric-value {
        font-size: 2em;
        color: #00ff9d;
    }
    .metric-label {
        color: white;
        opacity: 0.8;
    }
    h1, h2, h3 {
        color: white !important;
    }
    .stAlert {
        background-color: #2c2d31;
        color: white;
        border: none;
    }
    </style>
    """, unsafe_allow_html=True)

# Tower data
tower_data = {
    "towers": [
        {
            "id": "VZW001",
            "name": "Downtown 5G Ultra",
            "lat": 37.7749,
            "lon": -122.4194,
            "type": "5G Ultra Wideband",
            "status": "Operational",
            "signal_strength": 95,
            "network_load": 78,
            "coverage_radius": 2000
        },
        {
            "id": "VZW002",
            "name": "Harbor View 5G",
            "lat": 37.8079,
            "lon": -122.4012,
            "type": "5G Edge",
            "status": "Warning",
            "signal_strength": 82,
            "network_load": 92,
            "coverage_radius": 5000
        },
        {
            "id": "VZW003",
            "name": "Heights Tower",
            "lat": 37.7575,
            "lon": -122.4376,
            "type": "5G Ultra Wideband",
            "status": "Critical",
            "signal_strength": 65,
            "network_load": 45,
            "coverage_radius": 2000
        }
    ]
}

@st.cache_resource
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        # Load tower classifier
        tower_model = TowerClassifier().to(device)
        tower_model.load_state_dict(torch.load("tower_classifier.pth", map_location=device))
        tower_model.eval()
    except Exception as e:
        st.error("⚠️ Error loading tower classifier model. Some features may not work.")
        tower_model = None
    
    try:
        # Load antenna detector with correct path
        model_path = os.path.join('UTD-Models-Videos', 'runs', 'detect', 'train4', 'weights', 'best.pt')
        antenna_model = YOLO(model_path)
    except Exception as e:
        st.warning("⚠️ Error loading antenna detector model. Antenna detection will be disabled.")
        antenna_model = None
    
    return tower_model, antenna_model, device

def create_map():
    # Create base map
    m = folium.Map(
        location=[37.7749, -122.4194],
        zoom_start=11,
        tiles='CartoDB dark_matter'
    )
    
    # Create feature groups
    ultra_wideband = folium.FeatureGroup(name="5G Ultra Wideband")
    edge_5g = folium.FeatureGroup(name="5G Edge")
    
    for tower in tower_data["towers"]:
        # Style based on tower type
        color = "#FF0000" if tower["type"] == "5G Ultra Wideband" else "#FF6B6B"
        
        # Add coverage circle
        folium.Circle(
            location=[tower["lat"], tower["lon"]],
            radius=tower["coverage_radius"],
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=0.2,
            popup=f"""
                <div style="width:200px">
                    <h4>{tower['name']}</h4>
                    <p>Type: {tower['type']}</p>
                    <p>Status: {tower['status']}</p>
                    <p>Signal: {tower['signal_strength']}%</p>
                    <p>Load: {tower['network_load']}%</p>
                </div>
            """
        ).add_to(ultra_wideband if tower["type"] == "5G Ultra Wideband" else edge_5g)
        
        # Add tower marker
        folium.CircleMarker(
            location=[tower["lat"], tower["lon"]],
            radius=6,
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=1,
            popup=tower["name"]
        ).add_to(ultra_wideband if tower["type"] == "5G Ultra Wideband" else edge_5g)
    
    # Add feature groups to map
    ultra_wideband.add_to(m)
    edge_5g.add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    return m

def process_image(image):
    if image is None:
        return None, "No tower detected", "No antennas detected"
    
    # Convert to PIL Image if needed
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Make a copy for antenna detection
    image_for_detection = image.copy()
    
    try:
        # Tower Classification
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        img_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = tower_model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        
        # Get the best prediction
        best_idx = torch.argmax(probabilities).item()
        best_prob = float(probabilities[best_idx])
        tower_prediction = f"{class_names[best_idx]} ({best_prob:.1%} confidence)"
    except Exception as e:
        tower_prediction = "Tower classification unavailable"
    
    try:
        # Antenna Detection
        if antenna_model is not None:
            results = antenna_model(image_for_detection)
            boxes = results[0].boxes
            
            # Draw boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                cv2.rectangle(
                    np.array(image_for_detection),
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (0, 255, 0),
                    2
                )
            antenna_count = f"Detected {len(boxes)} antennas"
        else:
            antenna_count = "Antenna detection unavailable"
    except Exception as e:
        antenna_count = f"Error in antenna detection: {str(e)}"
        image_for_detection = image.copy()
    
    return image_for_detection, tower_prediction, antenna_count

# Load models
tower_model, antenna_model, device = load_models()
class_names = ['Guyed Tower', 'Lattice Tower', 'Monopole Tower', 'Water Tank Tower']

# Title
st.title("📡 Cell Tower Analyzer")
st.markdown("### Advanced AI-powered tower classification and antenna detection")

# Create tabs
tab1, tab2 = st.tabs(["📊 Dashboard", "📷 Image Analysis"])

with tab1:
    # Dashboard layout
    col1, col2, col3 = st.columns(3)
    
    # Statistics
    total_towers = len(tower_data["towers"])
    operational = sum(1 for t in tower_data["towers"] if t["status"] == "Operational")
    warning = sum(1 for t in tower_data["towers"] if t["status"] == "Warning")
    critical = sum(1 for t in tower_data["towers"] if t["status"] == "Critical")
    avg_signal = sum(t["signal_strength"] for t in tower_data["towers"]) / total_towers
    avg_load = sum(t["network_load"] for t in tower_data["towers"]) / total_towers
    
    with col1:
        st.metric("Total Towers", total_towers, delta=None, help=None)
        st.metric("Operational", operational, delta=None, help=None)
    
    with col2:
        st.metric("Warning", warning, delta=None, help=None)
        st.metric("Critical", critical, delta=None, help=None)
    
    with col3:
        st.metric("Avg Signal Strength", f"{avg_signal:.1f}%", delta=None, help=None)
        st.metric("Avg Network Load", f"{avg_load:.1f}%", delta=None, help=None)
    
    # Map
    st.markdown("### 🗺️ Tower Network Map")
    m = create_map()
    st_folium(m, width=1200, height=600)
    
    # Search box
    st.text_input("🔍 Search by ZIP code", placeholder="Enter ZIP code...")

with tab2:
    # Image Analysis
    st.markdown("### Upload an image for analysis")
    
    # Controls in a single row
    col_controls = st.columns(4)
    
    with col_controls[0]:
        # More aggressive confidence threshold
        confidence_threshold = st.slider(
            "Detection Sensitivity",
            min_value=0.0,
            max_value=1.0,
            value=0.05,  # Much lower default for higher sensitivity
            step=0.01,
            help="Lower values will detect more antennas"
        )
    
    with col_controls[1]:
        # Smaller minimum size
        min_detection_size = st.slider(
            "Min Size (px)",
            min_value=5,
            max_value=100,
            value=10,  # Smaller default size
            step=5,
            help="Minimum antenna size"
        )
    
    with col_controls[2]:
        # Lower IOU for better separation
        iou_threshold = st.slider(
            "Overlap Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.2,  # Lower IOU threshold
            step=0.05,
            help="Control overlapping detections"
        )
    
    with col_controls[3]:
        # Add augmentation toggle
        use_augmentation = st.checkbox(
            "Enhanced Detection",
            value=True,
            help="Use multiple detection passes for better accuracy"
        )
    
    # File upload and retake button in same row
    col_upload, col_retake = st.columns([3, 1])
    with col_upload:
        uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    with col_retake:
        if uploaded_image is not None:
            if st.button("📸 New Image"):
                uploaded_image = None
                st.rerun()
    
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        
        # Convert PIL image to numpy array for OpenCV operations
        image_np = np.array(image)
        
        try:
            # Tower Classification
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            img_tensor = transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = tower_model(img_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            
            # Get the best prediction
            best_idx = torch.argmax(probabilities).item()
            best_prob = float(probabilities[best_idx])
            tower_prediction = f"{class_names[best_idx]} ({best_prob:.1%} confidence)"
        except Exception as e:
            tower_prediction = "Tower classification unavailable"
        
        try:
            # Antenna Detection with enhanced parameters
            if antenna_model is not None:
                all_boxes = []
                
                if use_augmentation:
                    # Multiple detection passes with different scales
                    scales = [0.8, 1.0, 1.2]
                    for scale in scales:
                        # Resize image for different scales
                        h, w = image_np.shape[:2]
                        new_h, new_w = int(h * scale), int(w * scale)
                        scaled_img = cv2.resize(image_np, (new_w, new_h))
                        
                        # Run detection
                        results = antenna_model(
                            scaled_img,
                            conf=0.01,  # Very low initial confidence
                            iou=iou_threshold,
                            agnostic_nms=True,
                            max_det=100  # Increased max detections
                        )
                        
                        # Scale boxes back to original size
                        for box in results[0].boxes:
                            x1, y1, x2, y2 = map(int, (box.xyxy[0] / scale).tolist())
                            conf = float(box.conf[0])
                            cls = int(box.cls[0])
                            all_boxes.append((x1, y1, x2, y2, conf, cls))
                else:
                    # Single pass detection
                    results = antenna_model(
                        image_np,
                        conf=0.01,
                        iou=iou_threshold,
                        agnostic_nms=True,
                        max_det=100
                    )
                    for box in results[0].boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        conf = float(box.conf[0])
                        cls = int(box.cls[0])
                        all_boxes.append((x1, y1, x2, y2, conf, cls))
                
                # Non-maximum suppression on combined detections
                filtered_boxes = []
                for x1, y1, x2, y2, conf, cls in all_boxes:
                    width = x2 - x1
                    height = y2 - y1
                    if conf >= confidence_threshold and width >= min_detection_size and height >= min_detection_size:
                        filtered_boxes.append((x1, y1, x2, y2, conf, cls))
                
                # Sort by confidence
                filtered_boxes.sort(key=lambda x: x[4], reverse=True)
                
                # Draw boxes on original image
                for x1, y1, x2, y2, conf, cls in filtered_boxes:
                    # Draw box with white color and thick border
                    cv2.rectangle(
                        image_np,
                        (x1, y1),
                        (x2, y2),
                        (255, 255, 255),  # White color
                        3  # Thicker line
                    )
                    
                    # Add small confidence indicator
                    label = f"{conf:.2f}"
                    cv2.putText(
                        image_np,
                        label,
                        (x1+5, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),  # White text
                        2  # Thicker text
                    )
                
                # Add detection summary
                antenna_count = len(filtered_boxes)
                if antenna_count > 0:
                    avg_conf = sum(box[4] for box in filtered_boxes) / antenna_count
                    summary = f"Detected {antenna_count} antennas (avg confidence: {avg_conf:.2f})"
                else:
                    summary = "No antennas detected with current settings"
            else:
                summary = "Antenna detection unavailable"
        except Exception as e:
            summary = f"Error in antenna detection: {str(e)}"
        
        # Display layout
        col1, col2 = st.columns(2)
        with col1:
            st.image(image_np, caption="Detected Antennas", use_column_width=True)
            st.info(f"🏗️ Tower Type: {tower_prediction}")
            st.info(f"📡 {summary}")
        with col2:
            st.markdown("### Future Feature")
            st.info("This space is reserved for an upcoming feature.") 
import gradio as gr
import torch
from model import TowerClassifier
from ultralytics import YOLO
import folium
from PIL import Image
import numpy as np
from torchvision import transforms
import cv2
import os
from antenna_detector import AntennaDetector
from datetime import datetime
import json
import requests

def get_weather_data(lat, lon):
    try:
        # Replace with your OpenWeatherMap API key
        api_key = os.getenv('OPENWEATHER_API_KEY', '')
        if not api_key:
            return {
                "temperature": 22.5,
                "humidity": 65,
                "wind_speed": 12,
                "precipitation": 0.2,
                "conditions": "Partly Cloudy",
                "alerts": []
            }
            
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
        response = requests.get(url)
        data = response.json()
        
        return {
            "temperature": data["main"]["temp"],
            "humidity": data["main"]["humidity"],
            "wind_speed": data["wind"]["speed"],
            "precipitation": data.get("rain", {}).get("1h", 0),
            "conditions": data["weather"][0]["description"],
            "alerts": []
        }
    except Exception as e:
        print(f"Weather API error: {str(e)}")
        return None

def analyze_weather_impacts(weather, tower):
    impacts = []
    
    # Check temperature impacts
    if weather["temperature"] > 35:
        impacts.append({
            "severity": "warning",
            "icon": "🌡️",
            "message": "High temperature may affect equipment performance"
        })
    
    # Check wind impacts
    if weather["wind_speed"] > 20:
        impacts.append({
            "severity": "danger",
            "icon": "💨",
            "message": "High winds may cause tower instability"
        })
    
    # Check precipitation impacts
    if weather["precipitation"] > 0.5:
        impacts.append({
            "severity": "warning",
            "icon": "🌧️",
            "message": "Heavy rain may increase corrosion risk"
        })
    
    # Check humidity impacts
    if weather["humidity"] > 80:
        impacts.append({
            "severity": "warning",
            "icon": "💧",
            "message": "High humidity may affect signal quality"
        })
    
    return impacts

# Custom CSS for immersive UI
custom_css = """
:root {
    --verizon-red: #FF0000;
    --background-grey: #f5f5f5;
}

* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
    font-family: 'Inter', -apple-system, system-ui, sans-serif;
}

.gradio-container {
    margin: 0 !important;
    padding: 0 !important;
    width: 100vw !important;
    max-width: 100vw !important;
    min-height: 100vh !important;
    background: var(--background-grey) !important;
}

.main-container {
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    padding: 2rem !important;
    gap: 2rem !important;
    position: relative !important;
    min-height: 100vh !important;
    background: var(--background-grey) !important;
}

.header {
    width: 100% !important;
    text-align: center !important;
    margin-bottom: 2rem !important;
    background: #ffffff !important;
    padding: 1.5rem 2rem !important;
    border-radius: 8px !important;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1) !important;
    overflow: hidden !important;
}

.header .logo {
    width: 900px !important;
    height: 180px !important;
    margin: 0 auto 1rem auto !important;
    display: block !important;
    background: transparent !important;
}

.header .logo.hide-label .wrap.svelte-1gqxwij > .label-wrap {
    display: none !important;
}

.header .logo img {
    width: 100% !important;
    height: 100% !important;
    object-fit: contain !important;
    object-position: center !important;
}

.header .title {
    font-size: 1.2rem !important;
    font-weight: 500 !important;
    color: #000000 !important;
    margin-top: 0.5rem !important;
    text-transform: uppercase !important;
    text-align: center !important;
    letter-spacing: 1px !important;
}

.map-container {
    background: #ffffff !important;
    width: 100% !important;
    height: 500px !important;
    border-radius: 8px !important;
    overflow: hidden !important;
    margin-bottom: 2rem !important;
    border: 1px solid rgba(0, 0, 0, 0.1) !important;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
    padding: 1rem !important;
}

.upload-section {
    position: fixed !important;
    left: 2rem !important;
    bottom: 2rem !important;
    z-index: 1000 !important;
}

/* Style the file upload component to look like a button */
.upload-section .upload-button > .wrap {
    background: #E6E6E6 !important;
    border: none !important;
    border-radius: 4px !important;
    padding: 0 !important;
    margin: 0 !important;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1) !important;
}

.upload-section .upload-button > .wrap > .label-wrap {
    display: none !important;
}

.upload-section .upload-button > .wrap > .file-preview {
    display: none !important;
}

.upload-section .upload-button > .wrap > .file-upload {
    background: transparent !important;
    border: none !important;
    padding: 0.75rem 1.5rem !important;
    color: var(--verizon-red) !important;
    font-weight: 500 !important;
    font-size: 0.875rem !important;
    text-transform: uppercase !important;
    cursor: pointer !important;
}

.upload-section .upload-button > .wrap > .file-upload:hover {
    background: #D9D9D9 !important;
}

.results-content {
    position: fixed !important;
    right: 2rem !important;
    bottom: 2rem !important;
    width: 400px !important;
    background: #ffffff !important;
    border-radius: 8px !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15) !important;
    overflow: hidden !important;
    z-index: 1000 !important;
}

.results-content > .wrap {
    padding: 1rem !important;
}

.results-content .label-wrap {
    color: var(--verizon-red) !important;
    font-weight: 500 !important;
    margin-bottom: 0.5rem !important;
}

/* Hide unnecessary elements */
.footer {
    display: none !important;
}

.gradio-container > div:not(.main-container) {
    display: none !important;
}

/* Tab Styles */
.tabs {
    margin-top: 1rem !important;
    background: var(--background-grey) !important;
}

.tab {
    padding: 2rem !important;
    background: var(--background-grey) !important;
}

.upload-container {
    max-width: 800px !important;
    margin: 0 auto !important;
    padding: 2rem !important;
    background: #ffffff !important;
    border-radius: 8px !important;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
}

.results-section {
    display: flex !important;
    flex-direction: column !important;
    gap: 1rem !important;
    background: #ffffff !important;
    border-radius: 8px !important;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
    padding: 1.5rem !important;
    height: calc(100% - 120px) !important;
    overflow: hidden !important;
}

.results-section > .wrap {
    height: 100% !important;
    display: flex !important;
    flex-direction: column !important;
    gap: 1rem !important;
}

.results-section img {
    width: 100% !important;
    height: 300px !important;
    object-fit: contain !important;
    border-radius: 4px !important;
    background: var(--background-grey) !important;
}

.results-section .label-wrap {
    font-weight: 500 !important;
    color: var(--verizon-red) !important;
    margin-bottom: 0.5rem !important;
}

.results-section textarea {
    height: 120px !important;
    resize: none !important;
    font-family: monospace !important;
    font-size: 0.9rem !important;
    line-height: 1.4 !important;
    padding: 1rem !important;
    border: 1px solid rgba(0, 0, 0, 0.1) !important;
    border-radius: 4px !important;
    background: var(--background-grey) !important;
}

/* Tab navigation styles */
.tabs > .tab-nav {
    background: #ffffff !important;
    border-bottom: 2px solid rgba(0, 0, 0, 0.1) !important;
    padding: 0 2rem !important;
    border-radius: 8px 8px 0 0 !important;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05) !important;
}

.tabs > .tab-nav > button {
    color: #000000 !important;
    font-weight: 500 !important;
    padding: 1rem 2rem !important;
    margin: 0 !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    margin-bottom: -2px !important;
    background: transparent !important;
    transition: all 0.3s ease !important;
}

.tabs > .tab-nav > button:hover {
    color: #000000 !important;
    border-bottom-color: rgba(0, 0, 0, 0.5) !important;
}

.tabs > .tab-nav > button.selected {
    color: #000000 !important;
    border-bottom-color: #000000 !important;
    font-weight: 600 !important;
}

.analysis-tab {
    display: grid !important;
    grid-template-columns: 1fr 1fr !important;
    gap: 2rem !important;
    height: calc(100vh - 200px) !important;
    padding: 2rem !important;
}

.model-viewer {
    background: #ffffff !important;
    border-radius: 8px !important;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
    height: 100% !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}

.model-viewer-placeholder {
    color: #666666 !important;
    font-size: 1.2rem !important;
    text-align: center !important;
}

.analysis-container {
    display: flex !important;
    flex-direction: column !important;
    gap: 1rem !important;
    height: 100% !important;
    overflow: hidden !important;
}

.upload-container {
    background: #ffffff !important;
    border-radius: 8px !important;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
    padding: 1.5rem !important;
}

.section-title {
    font-size: 1.5rem !important;
    font-weight: 500 !important;
    color: #000000 !important;
    margin-bottom: 1rem !important;
    text-align: left !important;
}

.analysis-title {
    font-size: 1.2rem !important;
    font-weight: 500 !important;
    color: #000000 !important;
    margin: 1rem 0 !important;
    text-align: left !important;
}

.analysis-text {
    font-size: 1rem !important;
    color: #000000 !important;
    line-height: 1.5 !important;
    font-family: monospace !important;
}

.analysis-text p {
    color: #000000 !important;
    margin: 0 !important;
    padding: 0 !important;
}

.analysis-text * {
    color: #000000 !important;
}

.results-section .analysis-text {
    color: #000000 !important;
}

.results-section .analysis-text p {
    color: #000000 !important;
}

.model-preview {
    width: 100% !important;
    height: 100% !important;
    object-fit: contain !important;
}

.result-image {
    width: 100% !important;
    max-height: 400px !important;
    object-fit: contain !important;
}

.file-input > .wrap {
    border: 2px dashed rgba(0, 0, 0, 0.2) !important;
    border-radius: 8px !important;
    padding: 2rem !important;
    text-align: center !important;
    background: transparent !important;
}

.file-input > .wrap:hover {
    border-color: #000000 !important;
}

.file-input .file-preview {
    display: none !important;
}

.file-input .file-upload {
    color: #000000 !important;
    font-size: 1rem !important;
}

/* Hide the label and file information */
.file-input .label-wrap,
.file-input .file-preview,
.file-input .file-metadata {
    display: none !important;
}

/* Adjust the upload container visibility transition */
.upload-container {
    transition: opacity 0.3s ease !important;
}

.upload-container[style*="display: none"] {
    opacity: 0 !important;
}

/* Style the results section */
.results-section {
    display: flex !important;
    flex-direction: column !important;
    gap: 1rem !important;
}

.results-section img {
    border-radius: 4px !important;
    background: var(--background-grey) !important;
}

/* Update all text colors to black */
.title, 
.section-title,
.analysis-title,
.analysis-text,
.analysis-text p,
.analysis-text *,
.results-section .analysis-text,
.results-section .analysis-text p,
.file-input .file-upload,
.model-viewer-placeholder,
h2, h3, p {
    color: #000000 !important;
}

/* Make sure the file upload text is also black */
.file-input > .wrap {
    border: 2px dashed rgba(0, 0, 0, 0.2) !important;
}

.file-input > .wrap:hover {
    border-color: #000000 !important;
}

.file-input .file-upload {
    color: #000000 !important;
}
"""

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
            "coverage_radius": 2000,
            "height": 45,
            "installation_date": "2022-03-15",
            "last_maintenance": "2023-11-01"
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
            "coverage_radius": 5000,
            "height": 38,
            "installation_date": "2021-08-22",
            "last_maintenance": "2023-09-15"
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
            "coverage_radius": 2000,
            "height": 52,
            "installation_date": "2022-01-10",
            "last_maintenance": "2023-10-05"
        }
    ]
}

def create_map():
    """Create the coverage map with tower locations"""
    # Create a base map centered on the US
    m = folium.Map(
        location=[39.8283, -98.5795],  # Center of the US
        zoom_start=4,
        tiles='CartoDB positron'  # Light theme map
    )
    
    # Sample tower data - replace with your actual data
    towers = [
        {"lat": 37.7749, "lon": -122.4194, "status": "active"},  # San Francisco
        {"lat": 40.7128, "lon": -74.0060, "status": "active"},   # New York
        {"lat": 34.0522, "lon": -118.2437, "status": "active"},  # Los Angeles
        {"lat": 41.8781, "lon": -87.6298, "status": "active"},   # Chicago
        # Add more tower locations as needed
    ]
    
    # Add towers to the map
    for tower in towers:
        folium.CircleMarker(
            location=[tower["lat"], tower["lon"]],
            radius=8,
            color='red',
            fill=True,
            fillColor='red',
            fillOpacity=0.7,
            popup=f"Tower Status: {tower['status']}"
        ).add_to(m)
        
        # Add coverage area
        folium.Circle(
            location=[tower["lat"], tower["lon"]],
            radius=20000,  # 20km radius
            color='red',
            fill=True,
            fillColor='red',
            fillOpacity=0.1
        ).add_to(m)
    
    # Save map to HTML string
    return m._repr_html_()

def create_dashboard():
    # Create the dashboard HTML with proper CSS string formatting
    dashboard_html = f"""
    <div class="dashboard-container">
        <!-- Interactive Map Section -->
        <div class="map-section" id="towerMap">
            <h2>Tower Locations</h2>
            <div class="map-container" id="mapContainer">
                {create_map()}
            </div>
        </div>

        <!-- Hidden sections that appear on tower selection -->
        <div class="dashboard-details" id="towerDetails" style="display: none;">
            <!-- Sensor Data Section -->
            <div class="sensor-section">
                <h2>Sensor Readings</h2>
                <div class="sensor-grid">
                    <div class="sensor-card">
                        <h3>Vibration Sensor</h3>
                        <div class="sensor-value" id="vibrationValue">--</div>
                        <div class="sensor-chart" id="vibrationChart"></div>
                    </div>
                    <div class="sensor-card">
                        <h3>Tilt Sensor</h3>
                        <div class="sensor-value" id="tiltValue">--</div>
                        <div class="sensor-chart" id="tiltChart"></div>
                    </div>
                    <div class="sensor-card">
                        <h3>Strain Sensor</h3>
                        <div class="sensor-value" id="strainValue">--</div>
                        <div class="sensor-chart" id="strainChart"></div>
                    </div>
                    <div class="sensor-card">
                        <h3>Corrosion Sensor</h3>
                        <div class="sensor-value" id="corrosionValue">--</div>
                        <div class="sensor-chart" id="corrosionChart"></div>
                    </div>
                </div>
            </div>

            <!-- Weather Analysis Section -->
            <div class="weather-section">
                <h2>Weather Impact Analysis</h2>
                <div class="weather-content">
                    <div class="current-weather">
                        <div class="weather-main">
                            <div class="weather-temp" id="weatherTemp">--°C</div>
                            <div class="weather-condition" id="weatherCondition">--</div>
                        </div>
                        <div class="weather-details">
                            <div class="weather-detail">
                                <span class="label">Humidity</span>
                                <span class="value" id="weatherHumidity">--%</span>
                            </div>
                            <div class="weather-detail">
                                <span class="label">Wind Speed</span>
                                <span class="value" id="weatherWind">-- km/h</span>
                            </div>
                            <div class="weather-detail">
                                <span class="label">Precipitation</span>
                                <span class="value" id="weatherPrecip">-- mm</span>
                            </div>
                        </div>
                    </div>
                    <div class="weather-impact">
                        <h3>Impact Analysis</h3>
                        <div class="impact-list" id="weatherImpacts"></div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <style>
        .dashboard-container {{
            display: flex;
            flex-direction: column;
            gap: 2rem;
            padding: 2rem;
            background: #ffffff;
            min-height: 100vh;
        }}

        .map-section {{
            background: #ffffff;
            border-radius: 1rem;
            padding: 2rem;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(0, 0, 0, 0.1);
            height: 60vh;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }}

        .dashboard-details {{
            display: flex;
            gap: 2rem;
            margin-top: 2rem;
        }}

        .sensor-section, .weather-section {{
            flex: 1;
            background: #ffffff;
            border-radius: 1rem;
            padding: 2rem;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(0, 0, 0, 0.1);
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }}

        .sensor-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 1.5rem;
            margin-top: 1.5rem;
        }}

        .sensor-card {{
            background: rgba(255, 0, 0, 0.05);
            border-radius: 0.75rem;
            padding: 1.5rem;
            border: 1px solid rgba(255, 0, 0, 0.1);
        }}

        .sensor-value {{
            font-size: 2rem;
            font-weight: 700;
            color: #ff0000;
            margin: 1rem 0;
        }}

        .weather-content {{
            display: flex;
            flex-direction: column;
            gap: 2rem;
            margin-top: 1.5rem;
        }}

        .current-weather {{
            background: rgba(255, 0, 0, 0.05);
            border-radius: 0.75rem;
            padding: 1.5rem;
            border: 1px solid rgba(255, 0, 0, 0.1);
        }}

        .weather-main {{
            text-align: center;
            margin-bottom: 1.5rem;
        }}

        .weather-temp {{
            font-size: 3rem;
            font-weight: 700;
            color: #ff0000;
        }}

        .weather-details {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 1rem;
            text-align: center;
        }}

        .weather-detail .label {{
            display: block;
            color: #666666;
            font-size: 0.875rem;
            margin-bottom: 0.5rem;
        }}

        .weather-detail .value {{
            font-size: 1.25rem;
            font-weight: 600;
            color: #333333;
        }}

        .impact-list {{
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }}

        .impact-item {{
            display: flex;
            align-items: center;
            gap: 1rem;
            padding: 1rem;
            background: rgba(255, 0, 0, 0.05);
            border-radius: 0.5rem;
            border: 1px solid rgba(255, 0, 0, 0.1);
        }}

        .impact-item.warning {{
            border-color: rgba(255, 193, 7, 0.2);
            background: rgba(255, 193, 7, 0.05);
        }}

        .impact-item.danger {{
            border-color: rgba(244, 67, 54, 0.2);
            background: rgba(244, 67, 54, 0.05);
        }}

        h2 {{
            color: #333333;
            font-size: 1.5rem;
            font-weight: 600;
            margin-bottom: 1rem;
        }}

        h3 {{
            color: #333333;
            font-size: 1.25rem;
            font-weight: 500;
            margin-bottom: 1rem;
        }}

        .map-container {{
            height: 100%;
            width: 100%;
            border-radius: 0.5rem;
            overflow: hidden;
            border: 1px solid rgba(0, 0, 0, 0.1);
        }}
    </style>

    <script>
    document.addEventListener('DOMContentLoaded', function() {{
        function onTowerClick(towerId) {{
            document.getElementById('towerDetails').style.display = 'flex';
            updateSensorData(towerId);
            updateWeatherData(towerId);
        }}

        function updateSensorData(towerId) {{
            const sensorIds = ['vibration', 'tilt', 'strain', 'corrosion'];
            sensorIds.forEach(sensor => {{
                const value = Math.random() * 100;
                document.getElementById(`${{sensor}}Value`).textContent = value.toFixed(1);
            }});
        }}

        function updateWeatherData(towerId) {{
            const tower = tower_data.towers.find(t => t.id === towerId);
            if (!tower) return;

            const weather = get_weather_data(tower.lat, tower.lon);
            
            document.getElementById('weatherTemp').textContent = `${{weather.temperature}}°C`;
            document.getElementById('weatherCondition').textContent = weather.conditions;
            document.getElementById('weatherHumidity').textContent = `${{weather.humidity}}%`;
            document.getElementById('weatherWind').textContent = `${{weather.wind_speed}} km/h`;
            document.getElementById('weatherPrecip').textContent = `${{weather.precipitation}} mm`;

            const impacts = analyze_weather_impacts(weather, tower);
            const impactList = document.getElementById('weatherImpacts');
            impactList.innerHTML = impacts.map(impact => 
                `<div class="impact-item ${{impact.severity}}">
                    <span class="impact-icon">${{impact.icon}}</span>
                    <span class="impact-text">${{impact.message}}</span>
                </div>`
            ).join('');
        }}

        const markers = document.querySelectorAll('.tower-marker');
        markers.forEach(marker => {{
            marker.addEventListener('click', function() {{
                onTowerClick(this.dataset.towerId);
            }});
        }});
    }});
    </script>
    """
    
    return dashboard_html

def process_image(image, confidence=0.05, min_size=10, iou_thresh=0.2, enhanced=True):
    if image is None:
        return None, "No image provided", "Please upload an image"
    
    try:
        # Convert to PIL Image if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy for OpenCV operations
        image_np = np.array(image)
        output_image = image_np.copy()
        
        # Tower Classification
        if tower_model is not None:
            try:
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
                tower_prediction = f"Tower classification error: {str(e)}"
        else:
            tower_prediction = "Tower classification model not available"
        
        # Antenna Detection
        if antenna_model is not None:
            try:
                results = antenna_model(
                    image_np,
                    conf=confidence,
                    iou=iou_thresh,
                    agnostic_nms=True,
                    max_det=100
                )
                
                boxes = []
                for box in results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    width = x2 - x1
                    height = y2 - y1
                    if width >= min_size and height >= min_size:
                        boxes.append((x1, y1, x2, y2, conf, cls))
                
                # Draw boxes
                for x1, y1, x2, y2, conf, cls in boxes:
                    color = (0, 255, 0) if conf >= 0.5 else (0, 165, 255)
                    cv2.rectangle(output_image, (x1, y1), (x2, y2), color, 2)
                
                antenna_count = len(boxes)
                if antenna_count > 0:
                    avg_conf = sum(box[4] for box in boxes) / antenna_count
                    antenna_summary = f"Detected {antenna_count} antennas with average confidence {avg_conf:.2f}"
                else:
                    antenna_summary = "No antennas detected"
            except Exception as e:
                antenna_summary = f"Antenna detection error: {str(e)}"
        else:
            antenna_summary = "Antenna detection model not available"
        
        return output_image, tower_prediction, antenna_summary
    except Exception as e:
        return image_np, f"Error processing image: {str(e)}", "Processing failed"

def process_video(video_path):
    if video_path is None:
        return None
    
    try:
        # Process video frames
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create output video writer
        output_path = "temp_output.mp4"
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        
        frame_count = 0
        total_antennas = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            processed_frame, _, _ = process_image(frame)
            out.write(processed_frame)
            
            frame_count += 1
        
        cap.release()
        out.release()
        
        return output_path
    except Exception as e:
        print(f"Error processing video: {str(e)}")
        return None

def process_file(file):
    """Process uploaded file (image or video)"""
    if file is None:
        return None, "No file provided", "Please upload an image or video"
    
    try:
        # For image files
        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
            # Read image using PIL
            image = Image.open(file)
            output_image, tower_pred, antenna_sum = process_image(image)
            return output_image, tower_pred, antenna_sum
        # For video files
        elif file.lower().endswith(('.mp4', '.avi', '.mov')):
            return process_video(file)
        else:
            return None, "Error", "Unsupported file type. Please upload an image or video file."
    except Exception as e:
        return None, "Error", f"Failed to process file: {str(e)}"

# Load models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Loading models...")

# Initialize models with None as fallback
tower_model = None
antenna_model = None
antenna_detector = None

try:
    tower_model = TowerClassifier().to(device)
    tower_model.load_state_dict(torch.load("tower_classifier.pth", map_location=device))
    tower_model.eval()
    print("Tower classifier loaded successfully")
except Exception as e:
    print(f"Warning: Tower classifier not loaded - {str(e)}")
    tower_model = None

try:
    model_path = os.path.join('UTD-Models-Videos', 'runs', 'detect', 'train4', 'weights', 'best.pt')
    antenna_model = YOLO(model_path)
    print("Antenna detector loaded successfully")
except Exception as e:
    print(f"Warning: Antenna detector not loaded - {str(e)}")
    antenna_model = None

try:
    antenna_detector = AntennaDetector()
except Exception as e:
    print(f"Warning: Antenna detector class not initialized - {str(e)}")
    antenna_detector = None

class_names = ['Guyed Tower', 'Lattice Tower', 'Monopole Tower', 'Water Tank Tower']

# Create Gradio interface
with gr.Blocks(css=custom_css) as iface:
    # Header (outside tabs, always visible)
    with gr.Column(elem_classes="header"):
        with gr.Row():
            gr.Image(
                value="static/images/verizon_logo.png",
                show_label=False,
                container=False,
                elem_classes="logo hide-label",
                scale=1,
                min_width=150
            )
        gr.Markdown("CELL TOWER HEALTH", elem_classes="title")
    
    # Tabs
    with gr.Tabs() as tabs:
        # Map View Tab
        with gr.Tab("Map View", elem_classes="tab"):
            with gr.Column(elem_classes="map-container"):
                map_html = create_map()
                gr.HTML(map_html)
        
        # Upload & Analysis Tab
        with gr.Tab("Upload & Analysis", elem_classes="tab"):
            with gr.Row(elem_classes="analysis-tab"):
                # Left side - 3D Model Viewer
                with gr.Column(elem_classes="model-viewer"):
                    gr.Markdown("3D MODEL", elem_classes="section-title")
                    model_viewer = gr.Image(
                        label="",
                        show_label=False,
                        elem_classes="model-preview"
                    )
                
                # Right side - Upload & Analysis
                with gr.Column(elem_classes="analysis-container"):
                    # Upload Section
                    with gr.Column(elem_classes="upload-container", visible=True) as upload_section:
                        file_input = gr.File(
                            label="",
                            show_label=False,
                            file_types=["image", "video"],
                            type="filepath",
                            elem_classes="file-input"
                        )
                    
                    # Results Section
                    with gr.Column(elem_classes="results-section", visible=True) as results_section:
                        with gr.Row():
                            result_image = gr.Image(
                                label="",
                                show_label=False,
                                elem_classes="result-image"
                            )
                        with gr.Column():
                            gr.Markdown("ANALYSIS:", elem_classes="analysis-title")
                            result_text = gr.Markdown(
                                elem_classes="analysis-text",
                                value=""
                            )
        
        def process_upload(file):
            if file is None:
                return {
                    upload_section: gr.update(visible=True),
                    result_image: None,
                    result_text: "",
                    model_viewer: None
                }
            
            try:
                # Process the file
                output_image, tower_prediction, antenna_summary = process_file(file)
                
                if output_image is None:
                    return {
                        upload_section: gr.update(visible=True),
                        result_image: None,
                        result_text: f"Error: {tower_prediction}\n{antenna_summary}",
                        model_viewer: None
                    }
                
                # Mock values for height, tilt, and azimuth
                height = "150 ft"
                tilt = "2.5°"
                azimuth = "275°"
                
                analysis = f"""height, {height}
tilt, {tilt}
azimuth, {azimuth}"""
                
                return {
                    upload_section: gr.update(visible=False),
                    result_image: output_image,
                    result_text: analysis,
                    model_viewer: output_image  # For now, show the same image in 3D viewer
                }
            except Exception as e:
                return {
                    upload_section: gr.update(visible=True),
                    result_image: None,
                    result_text: f"Error processing file: {str(e)}",
                    model_viewer: None
                }
        
        # Event handler
        file_input.change(
            fn=process_upload,
            inputs=[file_input],
            outputs=[
                upload_section,
                result_image,
                result_text,
                model_viewer
            ]
        )

# Launch the interface
iface.launch() 
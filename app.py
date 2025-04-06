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
    background: #0a0c10 !important;
    color: #ffffff !important;
    background-image: linear-gradient(45deg, #0a0c10 0%, #1a1f2c 100%) !important;
}

.main-container {
    display: flex !important;
    flex-direction: column !important;
    min-height: 100vh !important;
    background: transparent !important;
}

.header {
    background: rgba(26, 31, 44, 0.8) !important;
    backdrop-filter: blur(10px) !important;
    -webkit-backdrop-filter: blur(10px) !important;
    border-bottom: 1px solid rgba(255, 255, 255, 0.1) !important;
    padding: 2rem !important;
    position: sticky !important;
    top: 0 !important;
    z-index: 100 !important;
    text-align: center !important;
}

.header h1 {
    font-size: 2.5rem !important;
    font-weight: 700 !important;
    color: #ffffff !important;
    margin: 0 !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    background: linear-gradient(120deg, #64b5f6, #1976d2) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    animation: glow 2s ease-in-out infinite alternate !important;
}

@keyframes glow {
    from {
        text-shadow: 0 0 10px rgba(100, 181, 246, 0.5),
                     0 0 20px rgba(100, 181, 246, 0.3);
    }
    to {
        text-shadow: 0 0 20px rgba(100, 181, 246, 0.7),
                     0 0 30px rgba(100, 181, 246, 0.5);
    }
}

.header p {
    color: rgba(255, 255, 255, 0.7) !important;
    margin-top: 1rem !important;
    font-size: 1rem !important;
    font-weight: 300 !important;
    letter-spacing: 1px !important;
}

.content {
    flex: 1 !important;
    padding: 3rem 2rem !important;
    max-width: 1400px !important;
    margin: 0 auto !important;
    width: 100% !important;
    position: relative !important;
}

.content::before {
    content: '' !important;
    position: absolute !important;
    top: 0 !important;
    left: 0 !important;
    right: 0 !important;
    bottom: 0 !important;
    background: radial-gradient(circle at center, rgba(100, 181, 246, 0.1) 0%, transparent 70%) !important;
    pointer-events: none !important;
}

.tabs {
    display: flex !important;
    gap: 1rem !important;
    margin-bottom: 3rem !important;
    border-bottom: 1px solid rgba(255, 255, 255, 0.1) !important;
    padding-bottom: 1rem !important;
    justify-content: center !important;
}

.tab-button {
    padding: 1rem 2rem !important;
    font-size: 1rem !important;
    font-weight: 500 !important;
    color: rgba(255, 255, 255, 0.7) !important;
    background: transparent !important;
    border: none !important;
    border-radius: 0.5rem !important;
    cursor: pointer !important;
    transition: all 0.3s ease !important;
    position: relative !important;
    overflow: hidden !important;
}

.tab-button::before {
    content: '' !important;
    position: absolute !important;
    top: 0 !important;
    left: 0 !important;
    right: 0 !important;
    bottom: 0 !important;
    background: rgba(100, 181, 246, 0.1) !important;
    transform: scaleX(0) !important;
    transform-origin: left !important;
    transition: transform 0.3s ease !important;
}

.tab-button:hover::before {
    transform: scaleX(1) !important;
}

.tab-button.active {
    color: #64b5f6 !important;
    background: rgba(100, 181, 246, 0.1) !important;
}

.upload-container {
    background: rgba(26, 31, 44, 0.6) !important;
    border-radius: 1rem !important;
    backdrop-filter: blur(10px) !important;
    -webkit-backdrop-filter: blur(10px) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    padding: 3rem !important;
    margin-bottom: 3rem !important;
    text-align: center !important;
    transition: transform 0.3s ease, box-shadow 0.3s ease !important;
}

.upload-container:hover {
    transform: translateY(-5px) !important;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3) !important;
}

.upload-area {
    border: 2px dashed rgba(100, 181, 246, 0.3) !important;
    border-radius: 1rem !important;
    padding: 4rem !important;
    text-align: center !important;
    background: rgba(100, 181, 246, 0.05) !important;
    cursor: pointer !important;
    transition: all 0.3s ease !important;
}

.upload-area:hover {
    border-color: #64b5f6 !important;
    background: rgba(100, 181, 246, 0.1) !important;
}

.upload-icon {
    width: 48px !important;
    height: 48px !important;
    margin-bottom: 1rem !important;
    color: #64748b !important;
}

.upload-text {
    color: #64748b !important;
    font-size: 0.875rem !important;
    margin-bottom: 0.5rem !important;
}

.settings-container {
    background: rgba(26, 31, 44, 0.6) !important;
    border-radius: 1rem !important;
    backdrop-filter: blur(10px) !important;
    -webkit-backdrop-filter: blur(10px) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    padding: 2rem !important;
    margin-bottom: 3rem !important;
}

.settings-header {
    display: flex !important;
    align-items: center !important;
    justify-content: space-between !important;
    margin-bottom: 1.5rem !important;
}

.settings-title {
    font-size: 1.25rem !important;
    font-weight: 600 !important;
    color: #ffffff !important;
    margin-bottom: 2rem !important;
    text-align: center !important;
}

.settings-grid {
    display: grid !important;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)) !important;
    gap: 2rem !important;
}

.slider-container {
    background: rgba(100, 181, 246, 0.05) !important;
    padding: 1.5rem !important;
    border-radius: 0.75rem !important;
    border: 1px solid rgba(100, 181, 246, 0.1) !important;
}

.slider-label {
    font-size: 0.875rem !important;
    color: rgba(255, 255, 255, 0.7) !important;
    margin-bottom: 1rem !important;
}

.analyze-button {
    background: linear-gradient(120deg, #64b5f6, #1976d2) !important;
    color: #ffffff !important;
    padding: 1rem 2rem !important;
    border-radius: 0.75rem !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    border: none !important;
    cursor: pointer !important;
    transition: all 0.3s ease !important;
    width: 100% !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
    position: relative !important;
    overflow: hidden !important;
}

.analyze-button::before {
    content: '' !important;
    position: absolute !important;
    top: 0 !important;
    left: 0 !important;
    width: 100% !important;
    height: 100% !important;
    background: linear-gradient(120deg, transparent, rgba(255, 255, 255, 0.2), transparent) !important;
    transform: translateX(-100%) !important;
    transition: transform 0.6s ease !important;
}

.analyze-button:hover::before {
    transform: translateX(100%) !important;
}

.results-container {
    background: rgba(26, 31, 44, 0.6) !important;
    border-radius: 1rem !important;
    backdrop-filter: blur(10px) !important;
    -webkit-backdrop-filter: blur(10px) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    padding: 2rem !important;
    animation: fadeIn 0.5s ease !important;
}

@keyframes fadeIn {
    from {
        opacity: 0;
        transform: translateY(20px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.results-header {
    margin-bottom: 1.5rem !important;
    padding-bottom: 1rem !important;
    border-bottom: 1px solid #e2e8f0 !important;
}

.results-title {
    font-size: 1.5rem !important;
    font-weight: 600 !important;
    color: #ffffff !important;
    margin-bottom: 2rem !important;
    text-align: center !important;
}

.results-grid {
    display: grid !important;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)) !important;
    gap: 2rem !important;
}

.metric-card {
    background: rgba(100, 181, 246, 0.05) !important;
    padding: 2rem !important;
    border-radius: 1rem !important;
    text-align: center !important;
    border: 1px solid rgba(100, 181, 246, 0.1) !important;
    transition: transform 0.3s ease !important;
}

.metric-card:hover {
    transform: translateY(-5px) !important;
}

.metric-value {
    font-size: 2rem !important;
    font-weight: 700 !important;
    color: #64b5f6 !important;
    margin-bottom: 1rem !important;
    background: linear-gradient(120deg, #64b5f6, #1976d2) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
}

.metric-label {
    font-size: 1rem !important;
    color: rgba(255, 255, 255, 0.7) !important;
    font-weight: 500 !important;
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
    # Create base map with dark theme
    m = folium.Map(
        location=[37.7749, -122.4194],
        zoom_start=12,
        tiles='CartoDB dark_matter'
    )
    
    # Add custom JavaScript for click events
    m.get_root().html.add_child(folium.Element("""
        <script>
        function onTowerClick(towerId) {
            // Show tower details
            document.getElementById('towerDetails').style.display = 'flex';
            
            // Trigger data updates
            updateSensorData(towerId);
            updateWeatherData(towerId);
        }
        </script>
    """))
    
    for tower in tower_data["towers"]:
        # Style based on status
        if tower["status"] == "Operational":
            color = "#4CAF50"  # Green
        elif tower["status"] == "Warning":
            color = "#FFC107"  # Yellow
        else:
            color = "#F44336"  # Red
        
        # Add coverage circle
        folium.Circle(
            location=[tower["lat"], tower["lon"]],
            radius=tower["coverage_radius"],
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=0.2,
            popup=f"""
                <div style="width:200px" class="tower-marker" data-tower-id="{tower['id']}"
                     onclick="onTowerClick('{tower['id']}')">
                    <h4>{tower['name']}</h4>
                    <p>Type: {tower['type']}</p>
                    <p>Status: {tower['status']}</p>
                    <p>Signal: {tower['signal_strength']}%</p>
                    <p>Load: {tower['network_load']}%</p>
                    <button onclick="onTowerClick('{tower['id']}')"
                            style="background: #4CAF50; color: white; border: none; 
                                   padding: 5px 10px; border-radius: 4px; 
                                   cursor: pointer; margin-top: 10px;">
                        View Details
                    </button>
                </div>
            """
        ).add_to(m)
        
        # Add tower marker
        folium.CircleMarker(
            location=[tower["lat"], tower["lon"]],
            radius=6,
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=1,
            popup=tower["name"],
            class_name=f"tower-marker"
        ).add_to(m)
    
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
            background: var(--main-bg);
            min-height: 100vh;
        }}

        .map-section {{
            background: rgba(26, 31, 44, 0.6);
            border-radius: 1rem;
            padding: 2rem;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            height: 60vh;
        }}

        .dashboard-details {{
            display: flex;
            gap: 2rem;
            margin-top: 2rem;
        }}

        .sensor-section, .weather-section {{
            flex: 1;
            background: rgba(26, 31, 44, 0.6);
            border-radius: 1rem;
            padding: 2rem;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }}

        .sensor-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 1.5rem;
            margin-top: 1.5rem;
        }}

        .sensor-card {{
            background: rgba(100, 181, 246, 0.05);
            border-radius: 0.75rem;
            padding: 1.5rem;
            border: 1px solid rgba(100, 181, 246, 0.1);
        }}

        .sensor-value {{
            font-size: 2rem;
            font-weight: 700;
            color: #64b5f6;
            margin: 1rem 0;
            background: linear-gradient(120deg, #64b5f6, #1976d2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}

        .weather-content {{
            display: flex;
            flex-direction: column;
            gap: 2rem;
            margin-top: 1.5rem;
        }}

        .current-weather {{
            background: rgba(100, 181, 246, 0.05);
            border-radius: 0.75rem;
            padding: 1.5rem;
            border: 1px solid rgba(100, 181, 246, 0.1);
        }}

        .weather-main {{
            text-align: center;
            margin-bottom: 1.5rem;
        }}

        .weather-temp {{
            font-size: 3rem;
            font-weight: 700;
            background: linear-gradient(120deg, #64b5f6, #1976d2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}

        .weather-details {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 1rem;
            text-align: center;
        }}

        .weather-detail .label {{
            display: block;
            color: rgba(255, 255, 255, 0.7);
            font-size: 0.875rem;
            margin-bottom: 0.5rem;
        }}

        .weather-detail .value {{
            font-size: 1.25rem;
            font-weight: 600;
            color: #ffffff;
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
            background: rgba(100, 181, 246, 0.05);
            border-radius: 0.5rem;
            border: 1px solid rgba(100, 181, 246, 0.1);
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
            color: #ffffff;
            font-size: 1.5rem;
            font-weight: 600;
            margin-bottom: 1rem;
        }}

        h3 {{
            color: #ffffff;
            font-size: 1.25rem;
            font-weight: 500;
            margin-bottom: 1rem;
        }}

        .map-container {{
            height: 100%;
            width: 100%;
            border-radius: 0.5rem;
            overflow: hidden;
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
                
                if enhanced:
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
                            iou=iou_thresh,
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
                        iou=iou_thresh,
                        agnostic_nms=True,
                        max_det=100
                    )
                    for box in results[0].boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        conf = float(box.conf[0])
                        cls = int(box.cls[0])
                        all_boxes.append((x1, y1, x2, y2, conf, cls))
                
                # Filter and sort boxes
                filtered_boxes = []
                for x1, y1, x2, y2, conf, cls in all_boxes:
                    width = x2 - x1
                    height = y2 - y1
                    if conf >= confidence and width >= min_size and height >= min_size:
                        filtered_boxes.append((x1, y1, x2, y2, conf, cls))
                
                filtered_boxes.sort(key=lambda x: x[4], reverse=True)
                
                # Draw boxes on image
                for x1, y1, x2, y2, conf, cls in filtered_boxes:
                    # Draw white outline for contrast
                    cv2.rectangle(
                        output_image,
                        (x1-2, y1-2),
                        (x2+2, y2+2),
                        (255, 255, 255),
                        4
                    )
                    # Draw colored box based on confidence
                    color = (0, 255, 0) if conf >= 0.5 else (0, 165, 255) if conf >= 0.3 else (0, 0, 255)
                    cv2.rectangle(
                        output_image,
                        (x1, y1),
                        (x2, y2),
                        color,
                        2
                    )
                    
                    # Add confidence score
                    label = f"{conf:.2f}"
                    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    cv2.rectangle(output_image, (x1, y1-20), (x1+w+10, y1), color, -1)
                    cv2.putText(
                        output_image,
                        label,
                        (x1+5, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        2
                    )
                
                # Create detection summary
                antenna_count = len(filtered_boxes)
                if antenna_count > 0:
                    avg_conf = sum(box[4] for box in filtered_boxes) / antenna_count
                    high_conf = sum(1 for box in filtered_boxes if box[4] >= 0.5)
                    med_conf = sum(1 for box in filtered_boxes if 0.3 <= box[4] < 0.5)
                    low_conf = sum(1 for box in filtered_boxes if box[4] < 0.3)
                    
                    antenna_summary = f"""Detected {antenna_count} antennas:
- High confidence (≥0.5): {high_conf}
- Medium confidence (0.3-0.5): {med_conf}
- Low confidence (<0.3): {low_conf}
Average confidence: {avg_conf:.2f}"""
                else:
                    antenna_summary = "No antennas detected with current settings"
            else:
                antenna_summary = "Antenna detection unavailable"
                output_image = image_np.copy()
        except Exception as e:
            antenna_summary = f"Error in antenna detection: {str(e)}"
            output_image = image_np.copy()
        
        return output_image, tower_prediction, antenna_summary
    except Exception as e:
        return None, f"Error processing image: {str(e)}", "Processing failed"

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

# Load models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Loading models...")

try:
    tower_model = TowerClassifier().to(device)
    tower_model.load_state_dict(torch.load("tower_classifier.pth", map_location=device))
    tower_model.eval()
    print("Tower classifier loaded successfully")
except Exception as e:
    print(f"Error loading tower classifier: {str(e)}")
    tower_model = None

try:
    model_path = os.path.join('UTD-Models-Videos', 'runs', 'detect', 'train4', 'weights', 'best.pt')
    antenna_model = YOLO(model_path)
    print("Antenna detector loaded successfully")
except Exception as e:
    print(f"Error loading antenna detector: {str(e)}")
    antenna_model = None

class_names = ['Guyed Tower', 'Lattice Tower', 'Monopole Tower', 'Water Tank Tower']

# Initialize models
antenna_detector = AntennaDetector()
tower_classifier = TowerClassifier()

# Create Gradio interface
with gr.Blocks(css=custom_css) as iface:
    with gr.Column(elem_classes="main-container"):
        # Header
        with gr.Column(elem_classes="header"):
            gr.Markdown("# Cell Tower Analyzer")
            gr.Markdown("Advanced AI-powered tower classification and antenna detection")
        
        # Main content
        with gr.Column(elem_classes="content"):
            # Tabs
            with gr.Tabs() as tabs:
                with gr.Tab("📊 Dashboard", elem_classes="tab"):
                    dashboard_html = gr.HTML(create_dashboard())
                
                with gr.Tab("📷 Image Analysis", elem_classes="tab"):
                    with gr.Column():
                        # Upload Section
                        with gr.Column(elem_classes="upload-container"):
                            gr.Markdown("### Upload Image")
                            image_input = gr.Image(type="numpy", label="")
                        
                        # Settings Section
                        with gr.Column(elem_classes="settings-container"):
                            with gr.Row(elem_classes="settings-header"):
                                gr.Markdown("### Detection Settings", elem_classes="settings-title")
                            
                            with gr.Column(elem_classes="settings-grid"):
                                confidence = gr.Slider(
                                    minimum=0.0, maximum=1.0, value=0.05, step=0.01,
                                    label="Detection Sensitivity",
                                    elem_classes="slider-container"
                                )
                                min_size = gr.Slider(
                                    minimum=5, maximum=100, value=10, step=5,
                                    label="Min Size (px)",
                                    elem_classes="slider-container"
                                )
                                iou_thresh = gr.Slider(
                                    minimum=0.0, maximum=1.0, value=0.2, step=0.05,
                                    label="Overlap Threshold",
                                    elem_classes="slider-container"
                                )
                                enhanced = gr.Checkbox(
                                    value=True,
                                    label="Enhanced Detection",
                                    info="Use multiple detection passes"
                                )
                        
                        # Analyze Button
                        detect_btn = gr.Button(
                            "Analyze Image",
                            elem_classes="analyze-button"
                        )
                        
                        # Results Section
                        with gr.Column(elem_classes="results-container", visible=False) as results_container:
                            with gr.Column(elem_classes="results-header"):
                                gr.Markdown("### Analysis Results", elem_classes="results-title")
                            
                            with gr.Column(elem_classes="results-grid"):
                                image_output = gr.Image(type="numpy", label="Detected Antennas")
                                with gr.Column(elem_classes="metric-card"):
                                    tower_output = gr.Textbox(label="Tower Classification")
                                with gr.Column(elem_classes="metric-card"):
                                    antenna_output = gr.Textbox(label="Detection Summary")
                
                with gr.Tab("🎥 Video Analysis", elem_classes="tab"):
                    with gr.Column():
                        # Upload Section
                        with gr.Column(elem_classes="upload-container"):
                            gr.Markdown("### Upload Video")
                            video_input = gr.Video(label="")
                        
                        # Process Button
                        process_btn = gr.Button(
                            "Process Video",
                            elem_classes="analyze-button"
                        )
                        
                        # Video Output
                        video_output = gr.Video(label="Processed Video")

    # Event handlers
    detect_btn.click(
        fn=process_image,
        inputs=[image_input, confidence, min_size, iou_thresh, enhanced],
        outputs=[image_output, tower_output, antenna_output],
        show_progress=True,
    ).then(
        fn=lambda: gr.update(visible=True),
        outputs=[results_container]
    )

    process_btn.click(
        fn=process_video,
        inputs=[video_input],
        outputs=[video_output],
        show_progress=True
    )

# Launch the interface
iface.launch(server_port=8501) 
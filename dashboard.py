import folium
from folium import plugins
import json
import random
from datetime import datetime, timedelta
import os

class TowerDashboard:
    def __init__(self):
        # Sample data - in a real app, this would come from a database
        self.tower_data = {
            "towers": [
                {
                    "id": "VZW001",
                    "name": "Downtown 5G Ultra",
                    "lat": 37.7749,
                    "lon": -122.4194,
                    "type": "5G Ultra Wideband",
                    "status": "Operational",
                    "last_maintenance": "2024-02-15",
                    "signal_strength": 95,
                    "antenna_count": 6,
                    "network_load": 78,
                    "coverage_radius": 2000  # meters
                },
                {
                    "id": "VZW002",
                    "name": "Harbor View 5G",
                    "lat": 37.8079,
                    "lon": -122.4012,
                    "type": "5G Edge",
                    "status": "Warning",
                    "last_maintenance": "2024-01-20",
                    "signal_strength": 82,
                    "antenna_count": 4,
                    "network_load": 92,
                    "coverage_radius": 5000  # meters
                },
                {
                    "id": "VZW003",
                    "name": "Heights Tower",
                    "lat": 37.7575,
                    "lon": -122.4376,
                    "type": "5G Ultra Wideband",
                    "status": "Critical",
                    "last_maintenance": "2023-12-10",
                    "signal_strength": 65,
                    "antenna_count": 5,
                    "network_load": 45,
                    "coverage_radius": 2000  # meters
                },
            ]
        }

    def _get_coverage_style(self, tower_type):
        return {
            "5G Ultra Wideband": {
                "color": "#FF0000",
                "fillColor": "#FF0000",
                "weight": 1,
                "fillOpacity": 0.2
            },
            "5G Edge": {
                "color": "#FF6B6B",
                "fillColor": "#FF6B6B",
                "weight": 1,
                "fillOpacity": 0.15
            },
            "4G LTE": {
                "color": "#FFA07A",
                "fillColor": "#FFA07A",
                "weight": 1,
                "fillOpacity": 0.1
            }
        }.get(tower_type, {
            "color": "#808080",
            "fillColor": "#808080",
            "weight": 1,
            "fillOpacity": 0.1
        })

    def _get_tower_icon(self, tower_type):
        if tower_type == "5G Ultra Wideband":
            return """
            <div style="font-size: 15px; color: white; background-color: red; 
                    border-radius: 50%; width: 20px; height: 20px; text-align: center; 
                    line-height: 20px; border: 2px solid white;">
                <i class="fa fa-broadcast-tower"></i>
            </div>
            """
        else:
            return """
            <div style="font-size: 15px; color: white; background-color: #FF6B6B; 
                    border-radius: 50%; width: 20px; height: 20px; text-align: center; 
                    line-height: 20px; border: 2px solid white;">
                <i class="fa fa-broadcast-tower"></i>
            </div>
            """

    def _create_popup_html(self, tower):
        # Calculate days since last maintenance
        last_maintenance = datetime.strptime(tower["last_maintenance"], "%Y-%m-%d")
        days_since = (datetime.now() - last_maintenance).days
        
        type_color = "#FF0000" if tower["type"] == "5G Ultra Wideband" else "#FF6B6B"
        
        return f"""
        <div style="width: 300px; padding: 15px; font-family: Arial, sans-serif; background: #1a1a1a; color: white; border-radius: 10px;">
            <h3 style="color: white; margin-bottom: 15px; border-bottom: 2px solid {type_color}; padding-bottom: 10px;">
                {tower["name"]}
            </h3>
            
            <div style="background: #2d2d2d; padding: 15px; border-radius: 8px; margin-bottom: 15px;">
                <div style="margin-bottom: 10px;">
                    <strong>Network Type:</strong> 
                    <span style="color: {type_color}">
                        {tower["type"]}
                    </span>
                </div>
                <div style="margin-bottom: 10px;">
                    <strong>Status:</strong> {tower["status"]}
                </div>
                <div style="margin-bottom: 10px;">
                    <strong>Last Maintenance:</strong> {tower["last_maintenance"]}
                    <br>
                    <small style="color: #888;">({days_since} days ago)</small>
                </div>
            </div>
            
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
                <div style="text-align: center; padding: 10px; background: #2d2d2d; border-radius: 8px; border-left: 3px solid #4CAF50;">
                    <div style="font-size: 24px; color: #4CAF50;">
                        {tower["signal_strength"]}%
                    </div>
                    <div style="color: #888; font-size: 12px;">
                        Signal Strength
                    </div>
                </div>
                
                <div style="text-align: center; padding: 10px; background: #2d2d2d; border-radius: 8px; border-left: 3px solid #2196F3;">
                    <div style="font-size: 24px; color: #2196F3;">
                        {tower["antenna_count"]}
                    </div>
                    <div style="color: #888; font-size: 12px;">
                        Antennas
                    </div>
                </div>
                
                <div style="text-align: center; padding: 10px; background: #2d2d2d; border-radius: 8px; border-left: 3px solid #FFC107; grid-column: span 2;">
                    <div style="font-size: 24px; color: #FFC107;">
                        {tower["network_load"]}%
                    </div>
                    <div style="color: #888; font-size: 12px;">
                        Network Load
                    </div>
                </div>
            </div>
        </div>
        """

    def generate_map(self):
        # Create a map centered on the first tower
        m = folium.Map(
            location=[self.tower_data["towers"][0]["lat"], self.tower_data["towers"][0]["lon"]],
            zoom_start=11,
            tiles='CartoDB dark_matter',  # Dark theme map
            attr='&copy; <a href="http://www.openstreetmap.org/copyright">OpenStreetMap</a> &copy; <a href="http://cartodb.com/attributions">CartoDB</a>'
        )

        # Create feature groups for different coverage types
        ultra_wideband = folium.FeatureGroup(name="5G Ultra Wideband")
        edge_5g = folium.FeatureGroup(name="5G Edge")
        
        # Add coverage areas and tower markers
        for tower in self.tower_data["towers"]:
            # Add coverage area
            style = self._get_coverage_style(tower["type"])
            folium.Circle(
                location=[tower["lat"], tower["lon"]],
                radius=tower["coverage_radius"],
                popup=folium.Popup(self._create_popup_html(tower), max_width=350),
                tooltip=f"{tower['name']} Coverage Area",
                **style
            ).add_to(ultra_wideband if tower["type"] == "5G Ultra Wideband" else edge_5g)

            # Add tower marker
            icon_html = self._get_tower_icon(tower["type"])
            icon = folium.DivIcon(html=icon_html)
            
            folium.Marker(
                location=[tower["lat"], tower["lon"]],
                icon=icon,
                tooltip=f"{tower['name']} ({tower['type']})"
            ).add_to(ultra_wideband if tower["type"] == "5G Ultra Wideband" else edge_5g)

        # Add the feature groups to the map
        ultra_wideband.add_to(m)
        edge_5g.add_to(m)

        # Add fullscreen control
        plugins.Fullscreen().add_to(m)
        
        # Add search box
        search_box = """
        <div style="position: absolute; top: 10px; left: 50px; z-index: 1000; background: white; padding: 10px; border-radius: 4px; box-shadow: 0 2px 4px rgba(0,0,0,0.2);">
            <input type="text" id="search-input" placeholder="Enter ZIP code..." 
                   style="padding: 8px; border: 1px solid #ccc; border-radius: 4px; width: 200px;">
            <button onclick="searchLocation()" 
                    style="padding: 8px 15px; background: #FF0000; color: white; border: none; border-radius: 4px; margin-left: 5px; cursor: pointer;">
                Search
            </button>
        </div>
        
        <script>
        function searchLocation() {
            var zip = document.getElementById('search-input').value;
            // In a real app, this would use a geocoding service
            alert('Searching for ZIP code: ' + zip);
        }
        </script>
        """
        m.get_root().html.add_child(folium.Element(search_box))

        # Add custom legend
        legend_html = """
        <div style="position: absolute; bottom: 30px; left: 30px; z-index: 1000; background: rgba(0,0,0,0.7); padding: 20px; border-radius: 10px;">
            <h4 style="color: white; margin-bottom: 15px;">Coverage Types</h4>
            <div style="margin-bottom: 10px;">
                <span style="display: inline-block; width: 20px; height: 20px; background: #FF0000; border-radius: 50%; margin-right: 10px;"></span>
                <span style="color: white;">5G Ultra Wideband</span>
            </div>
            <div style="margin-bottom: 10px;">
                <span style="display: inline-block; width: 20px; height: 20px; background: #FF6B6B; border-radius: 50%; margin-right: 10px;"></span>
                <span style="color: white;">5G Edge</span>
            </div>
            <div>
                <span style="display: inline-block; width: 20px; height: 20px; background: #FFA07A; border-radius: 50%; margin-right: 10px;"></span>
                <span style="color: white;">4G LTE</span>
            </div>
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))

        # Ensure the static directory exists
        static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
        os.makedirs(static_dir, exist_ok=True)
        
        # Save map to HTML in the static directory
        map_path = os.path.join(static_dir, 'tower_map.html')
        m.save(map_path)
        
        return './static/tower_map.html'

    def get_overall_stats(self):
        total_towers = len(self.tower_data["towers"])
        operational = sum(1 for t in self.tower_data["towers"] if t["status"] == "Operational")
        warning = sum(1 for t in self.tower_data["towers"] if t["status"] == "Warning")
        critical = sum(1 for t in self.tower_data["towers"] if t["status"] == "Critical")
        
        avg_signal = sum(t["signal_strength"] for t in self.tower_data["towers"]) / total_towers
        avg_load = sum(t["network_load"] for t in self.tower_data["towers"]) / total_towers
        
        return {
            "total_towers": total_towers,
            "operational": operational,
            "warning": warning,
            "critical": critical,
            "avg_signal": round(avg_signal, 1),
            "avg_load": round(avg_load, 1)
        } 
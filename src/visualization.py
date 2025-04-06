"""
Visualization module for displaying tower detections and analysis results
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Union, Any
import math

class Visualizer:
    """
    Visualization tools for tower detection and analysis
    """
    
    def __init__(self):
        """Initialize the visualizer"""
        # Font settings
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        self.font_thickness = 2
        
        # Default colors
        self.default_box_color = (0, 255, 0)  # Green
        self.text_color = (255, 255, 255)     # White
        self.highlight_color = (0, 0, 255)    # Red
        
        # Default detection colors (if not provided)
        self.default_class_colors = {
            'GSM Antenna': (0, 255, 0),        # Green
            'Microwave Antenna': (0, 0, 255),  # Blue
            'antenna': (255, 0, 0),            # Red
            'Lattice Tower': (255, 255, 0),    # Yellow
            'M Type Tower': (0, 255, 255)      # Cyan
        }
    
    def draw_detections(self, 
                       image: np.ndarray, 
                       detections: List[Dict[str, Any]],
                       class_colors: Dict[str, Tuple[int, int, int]] = None) -> np.ndarray:
        """
        Draw bounding boxes and labels on an image
        
        Args:
            image: Input image (BGR format)
            detections: List of detection dictionaries
            class_colors: Dictionary mapping class names to BGR color tuples
            
        Returns:
            image: Image with drawn bounding boxes and labels
        """
        if class_colors is None:
            class_colors = self.default_class_colors
            
        vis_image = image.copy()
        
        for det in detections:
            # Get bounding box coordinates
            if 'bbox' in det:
                if len(det['bbox']) == 4:
                    if isinstance(det['bbox'][2], int) and isinstance(det['bbox'][3], int):
                        # Format: [x1, y1, x2, y2]
                        x1, y1, x2, y2 = det['bbox']
                        w, h = x2 - x1, y2 - y1
                    else:
                        # Format: [x, y, w, h]
                        x1, y1, w, h = det['bbox']
                        x2, y2 = x1 + w, y1 + h
            else:
                continue
                
            # Get class name and confidence
            class_name = det.get('class_name', det.get('class', 'unknown'))
            confidence = det.get('confidence', 0.0)
            
            # Determine box color based on class
            if class_name in class_colors:
                box_color = class_colors[class_name]
            else:
                box_color = self.default_box_color
            
            # Draw bounding box
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), box_color, 2)
            
            # Draw label with confidence
            label = f"{class_name} {confidence:.2f}"
            
            # Measure text size for background rectangle
            (text_width, text_height), _ = cv2.getTextSize(
                label, self.font, self.font_scale, self.font_thickness
            )
            
            # Draw text background
            cv2.rectangle(
                vis_image, 
                (x1, y1 - text_height - 10), 
                (x1 + text_width + 10, y1), 
                box_color, 
                -1  # Filled rectangle
            )
            
            # Draw text
            cv2.putText(
                vis_image,
                label,
                (x1 + 5, y1 - 5),
                self.font,
                self.font_scale,
                self.text_color,
                self.font_thickness
            )
            
        return vis_image
    
    def draw_tower_analysis(self, 
                          image: np.ndarray,
                          detections: List[Dict[str, Any]],
                          analysis_results: Dict[str, Any],
                          class_colors: Dict[str, Tuple[int, int, int]] = None) -> np.ndarray:
        """
        Draw tower analysis visualization with insights
        
        Args:
            image: Input image (BGR format)
            detections: List of detection dictionaries
            analysis_results: Results from tower analysis
            class_colors: Dictionary mapping class names to BGR color tuples
            
        Returns:
            image: Image with drawn analysis visualization
        """
        # First draw the basic detections
        vis_image = self.draw_detections(image, detections, class_colors)
        
        # Create an info sidebar on the right side
        height, width, _ = vis_image.shape
        sidebar_width = 400
        
        # Create a canvas with extra width for the sidebar
        canvas = np.zeros((height, width + sidebar_width, 3), dtype=np.uint8)
        
        # Copy the visualization image to the canvas
        canvas[:, :width] = vis_image
        
        # Fill the sidebar with a dark background
        canvas[:, width:] = (30, 30, 30)  # Dark gray
        
        # Draw a vertical line separating the image and sidebar
        cv2.line(canvas, (width, 0), (width, height), (200, 200, 200), 2)
        
        # Add a title to the sidebar
        cv2.putText(
            canvas,
            "Tower Analysis",
            (width + 20, 40),
            self.font,
            1.2,
            (255, 255, 255),
            2
        )
        
        # Add tower type
        tower_type = analysis_results.get('tower_type', 'Unknown')
        cv2.putText(
            canvas,
            f"Type: {tower_type}",
            (width + 20, 90),
            self.font,
            0.8,
            (200, 200, 200),
            2
        )
        
        # Add height estimate
        height_estimate = analysis_results.get('height_estimate', 0)
        cv2.putText(
            canvas,
            f"Est. Height: {height_estimate:.1f} ft",
            (width + 20, 130),
            self.font,
            0.8,
            (200, 200, 200),
            2
        )
        
        # Add antenna counts
        antenna_counts = analysis_results.get('antenna_counts', {})
        y_pos = 170
        cv2.putText(
            canvas,
            "Antenna Counts:",
            (width + 20, y_pos),
            self.font,
            0.8,
            (200, 200, 200),
            2
        )
        y_pos += 35
        
        for antenna_type, count in antenna_counts.items():
            cv2.putText(
                canvas,
                f"  - {antenna_type}: {count}",
                (width + 30, y_pos),
                self.font,
                0.7,
                (180, 180, 180),
                1
            )
            y_pos += 30
        
        # Add insights header
        y_pos += 20
        cv2.putText(
            canvas,
            "Insights:",
            (width + 20, y_pos),
            self.font,
            0.8,
            (255, 200, 0),  # Orange-yellow
            2
        )
        y_pos += 35
        
        # Add insights
        insights = analysis_results.get('insights', [])
        for insight in insights:
            # Split long insights into multiple lines
            words = insight.split()
            lines = []
            current_line = []
            current_length = 0
            
            # Character limit per line (approximate)
            char_limit = 40
            
            for word in words:
                if current_length + len(word) + 1 <= char_limit:
                    current_line.append(word)
                    current_length += len(word) + 1
                else:
                    lines.append(' '.join(current_line))
                    current_line = [word]
                    current_length = len(word)
            
            if current_line:
                lines.append(' '.join(current_line))
            
            for line in lines:
                cv2.putText(
                    canvas,
                    line,
                    (width + 30, y_pos),
                    self.font,
                    0.6,
                    (160, 200, 255),  # Light blue
                    1
                )
                y_pos += 25
            
            # Add some space between insights
            y_pos += 5
        
        # Add coverage information if available
        if 'coverage_quality' in analysis_results or 'theoretical_radius_miles' in analysis_results:
            y_pos += 20
            cv2.putText(
                canvas,
                "Coverage Analysis:",
                (width + 20, y_pos),
                self.font,
                0.8,
                (0, 255, 200),  # Cyan-green
                2
            )
            y_pos += 35
            
            if 'theoretical_radius_miles' in analysis_results:
                radius = analysis_results.get('theoretical_radius_miles', 0)
                cv2.putText(
                    canvas,
                    f"  Theoretical Coverage: {radius:.2f} miles",
                    (width + 30, y_pos),
                    self.font,
                    0.6,
                    (0, 240, 160),
                    1
                )
                y_pos += 25
            
            if 'coverage_quality' in analysis_results:
                quality = analysis_results.get('coverage_quality', 'Unknown')
                cv2.putText(
                    canvas,
                    f"  Quality: {quality}",
                    (width + 30, y_pos),
                    self.font,
                    0.6,
                    (0, 240, 160),
                    1
                )
        
        return canvas
    
    def create_summary_image(self, 
                           image: np.ndarray, 
                           detections: List[Dict[str, Any]],
                           analysis_results: Dict[str, Any]) -> np.ndarray:
        """
        Create a summary image with detection boxes and basic information overlay
        
        Args:
            image: Input image
            detections: List of detection dictionaries
            analysis_results: Results from tower analysis
            
        Returns:
            summary_image: Image with summary overlay
        """
        vis_image = image.copy()
        
        # Draw detection boxes
        for det in detections:
            if 'bbox' in det:
                # Get bounding box coordinates
                if len(det['bbox']) == 4:
                    if isinstance(det['bbox'][2], int) and isinstance(det['bbox'][3], int):
                        # Format: [x1, y1, x2, y2]
                        x1, y1, x2, y2 = det['bbox']
                    else:
                        # Format: [x, y, w, h]
                        x1, y1, w, h = det['bbox']
                        x2, y2 = x1 + w, y1 + h
                    
                    # Get class name and confidence
                    class_name = det.get('class_name', det.get('class', 'unknown'))
                    confidence = det.get('confidence', 0.0)
                    
                    # Determine color based on class name
                    if class_name in self.default_class_colors:
                        color = self.default_class_colors[class_name]
                    else:
                        color = self.default_box_color
                    
                    # Draw bounding box
                    cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw label
                    label = f"{class_name} {confidence:.2f}"
                    cv2.putText(vis_image, label, (x1, y1 - 10), self.font, 0.5, color, 2)
        
        # Add summary information overlay
        info = []
        
        if "tower_type" in analysis_results:
            info.append(f"Tower: {analysis_results['tower_type']}")
            
        if "height_estimate" in analysis_results:
            info.append(f"Height: {analysis_results['height_estimate']:.1f}ft")
            
        if "total_antennas" in analysis_results:
            info.append(f"Antennas: {analysis_results['total_antennas']}")
            
        # Draw the information
        y_pos = 30
        for i, text in enumerate(info):
            cv2.putText(vis_image, 
                       text, 
                       (20, y_pos), 
                       self.font, 
                       0.8,  # Larger font
                       (255, 255, 255),
                       2)
            y_pos += 30
            
        return vis_image
    
    def create_comparison_grid(self, 
                             images: List[np.ndarray], 
                             titles: List[str] = None) -> np.ndarray:
        """
        Create a grid of images for comparison
        
        Args:
            images: List of images to display in a grid
            titles: List of titles for each image
            
        Returns:
            grid_image: Combined grid image
        """
        if not images:
            return np.zeros((400, 600, 3), dtype=np.uint8)
        
        # Determine grid size
        n_images = len(images)
        cols = min(4, n_images)
        rows = math.ceil(n_images / cols)
        
        # Resize images to a uniform size
        height, width = 400, 600  # Default size for each cell
        
        # Create blank canvas
        grid = np.zeros((rows * height, cols * width, 3), dtype=np.uint8)
        
        # Place images in grid
        for i, img in enumerate(images):
            if img is None:
                continue
                
            r, c = i // cols, i % cols
            
            # Resize image to fit the cell
            resized = cv2.resize(img, (width, height))
            
            # Place in grid
            grid[r*height:(r+1)*height, c*width:(c+1)*width] = resized
            
            # Add title if provided
            if titles and i < len(titles):
                cv2.putText(
                    grid,
                    titles[i],
                    (c*width + 10, r*height + 30),
                    self.font,
                    0.8,
                    (255, 255, 255),
                    2
                )
        
        return grid
    
    def plot_coverage_map(self, 
                        center_point: Tuple[float, float], 
                        radius_miles: float, 
                        background_image: np.ndarray = None) -> plt.Figure:
        """
        Generate a coverage map visualization
        
        Args:
            center_point: (latitude, longitude) of tower location
            radius_miles: Coverage radius in miles
            background_image: Optional background satellite image
            
        Returns:
            fig: Matplotlib figure with coverage map
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if background_image is not None:
            ax.imshow(background_image, extent=[-radius_miles, radius_miles, -radius_miles, radius_miles])
        
        # Draw coverage circle
        circle = plt.Circle((0, 0), radius_miles, color='b', fill=True, alpha=0.3)
        ax.add_patch(circle)
        
        # Add tower location
        ax.plot(0, 0, 'r^', markersize=12, label='Tower')
        
        # Set axis limits
        ax.set_xlim(-radius_miles * 1.2, radius_miles * 1.2)
        ax.set_ylim(-radius_miles * 1.2, radius_miles * 1.2)
        
        # Add labels and title
        ax.set_xlabel('Miles (East-West)')
        ax.set_ylabel('Miles (North-South)')
        ax.set_title(f'Estimated Coverage Area: {radius_miles:.2f} mile radius')
        
        # Add grid and legend
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()
        
        # Make equal aspect ratio
        ax.set_aspect('equal')
        
        return fig 
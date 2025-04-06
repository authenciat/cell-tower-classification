"""
Tower analysis module to extract insights from detected tower components
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import math

class TowerAnalyzer:
    """
    Analyzer for cell tower detections that identifies key characteristics
    and provides insights about the tower
    """
    
    def __init__(self):
        """Initialize the tower analyzer"""
        # Reference heights for different tower types (in feet)
        self.reference_heights = {
            'Lattice Tower': 200.0,  # Typical lattice tower height
            'M Type Tower': 150.0,   # Typical monopole height
            'GSM Antenna': 6.0,      # Typical GSM antenna height
            'Microwave Antenna': 4.0, # Typical microwave dish diameter
            'antenna': 5.0          # Generic antenna height
        }
        
        # Pixel-to-height ratio calibration (will be estimated from image)
        self.px_height_ratio = None
    
    def analyze(self, detections: List[Dict[str, Any]], image: np.ndarray) -> Dict[str, Any]:
        """
        Analyze tower detection results to provide insights
        
        Args:
            detections: List of detection dictionaries
            image: Original image
            
        Returns:
            analysis_results: Dictionary with analysis results
        """
        if not detections:
            return {
                'tower_type': 'Unknown',
                'height_estimate': 0,
                'antenna_count': 0,
                'obstruction_detected': False,
                'insights': ['No tower detected in the image']
            }
        
        # Extract tower type
        tower_type = self._identify_tower_type(detections)
        
        # Estimate tower height
        height_estimate = self._estimate_tower_height(detections, image)
        
        # Count antennas by type
        antenna_counts = self._count_antennas_by_type(detections)
        
        # Check for obstructions
        obstructions = self._identify_obstructions(detections, image)
        
        # Generate insights
        insights = self._generate_insights(tower_type, height_estimate, 
                                         antenna_counts, obstructions)
        
        # Compile results
        analysis_results = {
            'tower_type': tower_type,
            'height_estimate': height_estimate,
            'antenna_counts': antenna_counts,
            'total_antennas': sum(antenna_counts.values()),
            'obstructions': obstructions,
            'insights': insights
        }
        
        return analysis_results
    
    def _identify_tower_type(self, detections: List[Dict[str, Any]]) -> str:
        """Identify the type of tower from detections"""
        # Look for specific tower types
        for det in detections:
            if det['class_name'] == 'Lattice Tower':
                return 'Lattice Tower'
            elif det['class_name'] == 'M Type Tower':
                return 'Monopole Tower'
        
        # If no specific tower type is detected, make an estimate based on antenna arrangement
        antenna_count = sum(1 for det in detections if 'antenna' in det['class_name'].lower())
        
        if antenna_count > 0:
            return 'Unknown Tower Structure'
        else:
            return 'No Tower Detected'
    
    def _estimate_tower_height(self, detections: List[Dict[str, Any]], image: np.ndarray) -> float:
        """Estimate the height of the tower in feet"""
        # Find the tower structure detection
        tower_detection = None
        for det in detections:
            if det['class_name'] in ['Lattice Tower', 'M Type Tower']:
                tower_detection = det
                break
        
        if tower_detection is None:
            # No tower structure detected, estimate from antenna positions
            antenna_detections = [det for det in detections if 'antenna' in det['class_name'].lower()]
            
            if not antenna_detections:
                return 0.0
            
            # Find the lowest and highest antenna points
            min_y = min(det['bbox'][1] for det in antenna_detections)
            max_y = max(det['bbox'][3] for det in antenna_detections)
            
            # Rough estimate based on typical heights
            pixel_height = max_y - min_y
            ratio = pixel_height / image.shape[0]  # Normalized by image height
            
            # Rough estimate based on typical tower heights
            estimated_height = ratio * 200.0  # Assuming max normalized height corresponds to 200ft
            
            return estimated_height
        else:
            # Tower structure detected, use its dimensions
            x1, y1, x2, y2 = tower_detection['bbox']
            tower_pixel_height = y2 - y1
            
            # Estimate height based on reference values
            tower_type = tower_detection['class_name']
            reference_height = self.reference_heights.get(tower_type, 150.0)
            
            # Calculate based on how much of the tower is visible in the frame
            # Assuming the image captures most of the tower
            frame_height_ratio = tower_pixel_height / image.shape[0]
            estimated_height = reference_height * min(1.0, 1.0/frame_height_ratio)
            
            return estimated_height
    
    def _count_antennas_by_type(self, detections: List[Dict[str, Any]]) -> Dict[str, int]:
        """Count antennas by type"""
        antenna_counts = {
            'GSM Antenna': 0,
            'Microwave Antenna': 0,
            'Other Antenna': 0
        }
        
        for det in detections:
            if det['class_name'] == 'GSM Antenna':
                antenna_counts['GSM Antenna'] += 1
            elif det['class_name'] == 'Microwave Antenna':
                antenna_counts['Microwave Antenna'] += 1
            elif det['class_name'] == 'antenna':
                antenna_counts['Other Antenna'] += 1
        
        return antenna_counts
    
    def _identify_obstructions(self, detections: List[Dict[str, Any]], image: np.ndarray) -> List[Dict[str, Any]]:
        """Identify potential obstructions like bird nests"""
        # This would require additional detection classes for obstructions
        # For now, we can use a simple approach to identify unusual objects near antennas
        
        # Get antenna positions
        antenna_detections = [det for det in detections if 'antenna' in det['class_name'].lower()]
        
        # In a real system, we would analyze the image around these antennas
        # For now, return an empty list
        return []
    
    def _generate_insights(self, tower_type: str, height_estimate: float, 
                         antenna_counts: Dict[str, int], 
                         obstructions: List[Dict[str, Any]]) -> List[str]:
        """Generate insights based on analysis"""
        insights = []
        
        # Tower type insights
        if tower_type == 'Lattice Tower':
            insights.append("Lattice tower identified - suitable for high load capacity with multiple antennas.")
        elif tower_type == 'Monopole Tower':
            insights.append("Monopole tower identified - typically has cleaner aesthetics but lower capacity.")
        
        # Height insights
        if height_estimate > 150:
            insights.append(f"Tall tower structure (est. {height_estimate:.1f}ft) - good for wide area coverage.")
        elif height_estimate > 0:
            insights.append(f"Tower height estimated at {height_estimate:.1f}ft.")
        
        # Antenna insights
        total_antennas = sum(antenna_counts.values())
        if total_antennas > 5:
            insights.append(f"High antenna density ({total_antennas} antennas) may indicate good coverage capabilities.")
        elif total_antennas > 0:
            insights.append(f"Detected {total_antennas} antennas on the structure.")
        
        # Specific antenna types
        if antenna_counts['GSM Antenna'] > 0:
            insights.append(f"GSM cellular antennas ({antenna_counts['GSM Antenna']}) provide cellular coverage.")
        
        if antenna_counts['Microwave Antenna'] > 0:
            insights.append(f"Microwave antennas ({antenna_counts['Microwave Antenna']}) provide backhaul connectivity.")
        
        # Obstruction insights
        if obstructions:
            insights.append(f"Detected {len(obstructions)} potential obstructions that may affect signal quality.")
        
        # If no meaningful insights, provide a default message
        if not insights:
            insights.append("Not enough tower features detected for detailed analysis.")
        
        return insights
    
    def analyze_coverage(self, detections: List[Dict[str, Any]], image: np.ndarray) -> Dict[str, Any]:
        """
        Analyze potential coverage based on tower height and antenna configuration
        
        Args:
            detections: List of detection dictionaries
            image: Original image
            
        Returns:
            coverage_analysis: Dictionary with coverage analysis
        """
        # Basic coverage analysis
        tower_type = self._identify_tower_type(detections)
        height_estimate = self._estimate_tower_height(detections, image)
        antenna_counts = self._count_antennas_by_type(detections)
        
        # Calculate theoretical coverage radius based on tower height
        # Using a simplified model: coverage radius ≈ √(2 * Earth radius * tower height)
        earth_radius = 3959 * 5280  # Earth radius in feet
        if height_estimate > 0:
            coverage_radius_miles = math.sqrt(2 * earth_radius * height_estimate) / 5280
        else:
            coverage_radius_miles = 0
        
        # Adjust based on antenna types
        gsm_factor = 1.0 + (0.1 * antenna_counts['GSM Antenna'])
        
        # Calculate estimated coverage
        estimated_coverage = coverage_radius_miles * gsm_factor
        
        coverage_analysis = {
            'tower_height_ft': height_estimate,
            'theoretical_radius_miles': coverage_radius_miles,
            'estimated_coverage_miles': estimated_coverage,
            'coverage_quality': self._estimate_coverage_quality(antenna_counts)
        }
        
        return coverage_analysis
    
    def _estimate_coverage_quality(self, antenna_counts: Dict[str, int]) -> str:
        """Estimate the quality of coverage based on antenna configuration"""
        total_antennas = sum(antenna_counts.values())
        
        if total_antennas > 8:
            return "Excellent"
        elif total_antennas > 5:
            return "Good"
        elif total_antennas > 3:
            return "Moderate"
        elif total_antennas > 0:
            return "Limited"
        else:
            return "Unknown"

    def process_video(self, video_path: str) -> Dict[str, Any]:
        """
        Process a video file for tower analysis
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Analysis results combining information from multiple frames
        """
        # This would be implemented with VideoProcessor
        # Here we provide a stub for framework completeness
        
        # In a real implementation, we would:
        # 1. Sample frames from the video
        # 2. Run detection on each frame
        # 3. Analyze each frame
        # 4. Combine results from all frames
        
        # For now, just return a placeholder
        return {
            'error': 'Video processing not fully implemented'
        }

    def combine_frame_analyses(self, frame_analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Combine analyses from multiple video frames into one comprehensive analysis
        
        Args:
            frame_analyses: List of analysis results from individual frames
            
        Returns:
            Combined analysis results
        """
        if not frame_analyses:
            return {
                'tower_type': 'Unknown',
                'height_estimate': 0,
                'antenna_counts': {'GSM Antenna': 0, 'Microwave Antenna': 0, 'Other Antenna': 0},
                'total_antennas': 0,
                'insights': ['No analysis data available']
            }
        
        # Get the most common tower type
        tower_types = [a.get('tower_type', 'Unknown') for a in frame_analyses if a]
        if tower_types:
            from collections import Counter
            tower_type_counts = Counter(tower_types)
            most_common_type = tower_type_counts.most_common(1)[0][0]
        else:
            most_common_type = 'Unknown'
        
        # Get the average height estimate, excluding zeros
        height_estimates = [a.get('height_estimate', 0) for a in frame_analyses if a and a.get('height_estimate', 0) > 0]
        avg_height = sum(height_estimates) / len(height_estimates) if height_estimates else 0
        
        # Take the MAXIMUM antenna counts per type across all frames
        antenna_counts = {
            'GSM Antenna': 0,
            'Microwave Antenna': 0,
            'Other Antenna': 0
        }
        
        # Find which frame has the maximum total antennas for visualization
        max_antennas_frame_idx = -1
        max_antennas_count = 0
        
        for i, analysis in enumerate(frame_analyses):
            if not analysis or 'antenna_counts' not in analysis:
                continue
            
            total_in_frame = sum(analysis['antenna_counts'].values())
            if total_in_frame > max_antennas_count:
                max_antennas_count = total_in_frame
                max_antennas_frame_idx = i
            
            for antenna_type, count in analysis['antenna_counts'].items():
                antenna_counts[antenna_type] = max(antenna_counts[antenna_type], count)
        
        # Calculate total antennas based on maximum counts
        total_antennas = sum(antenna_counts.values())
        
        # Store the frame index that had max antennas for visualization
        best_frame_info = None
        if max_antennas_frame_idx >= 0 and max_antennas_frame_idx < len(frame_analyses):
            best_frame_info = {
                'frame_index': max_antennas_frame_idx,
                'antenna_count': max_antennas_count
            }
        
        # Intelligently filter and deduplicate insights
        all_insights = []
        filtered_insights = []
        height_insight_added = False
        antenna_count_insight_added = False
        microwave_insight_added = False
        gsm_insight_added = False
        
        # First gather all insights
        for analysis in frame_analyses:
            if analysis and 'insights' in analysis:
                all_insights.extend(analysis['insights'])
        
        # Apply smart filtering to remove redundant insights
        for insight in all_insights:
            # Skip redundant height insights
            if "Tower height estimated at" in insight or "Tall tower structure (est." in insight:
                if not height_insight_added:
                    # Just add one insight about the height using the average
                    if avg_height > 150:
                        filtered_insights.append(f"Tall tower structure (est. {avg_height:.1f}ft) - good for wide area coverage.")
                    else:
                        filtered_insights.append(f"Tower height estimated at {avg_height:.1f}ft.")
                    height_insight_added = True
                continue
            
            # Skip redundant antenna count insights
            if "Detected" in insight and "antennas on the structure" in insight:
                if not antenna_count_insight_added:
                    if total_antennas > 5:
                        filtered_insights.append(f"High antenna density ({total_antennas} antennas) may indicate good coverage capabilities.")
                    else:
                        filtered_insights.append(f"Detected {total_antennas} antennas on the structure.")
                    antenna_count_insight_added = True
                continue
                
            # Skip redundant microwave antenna insights
            if "Microwave antennas" in insight and "provide backhaul connectivity" in insight:
                if not microwave_insight_added:
                    microwave_count = antenna_counts.get('Microwave Antenna', 0)
                    if microwave_count > 0:
                        # Add the insight with the correct count
                        filtered_insights.append(f"Microwave antennas ({microwave_count}) provide backhaul connectivity.")
                    microwave_insight_added = True
                continue
            
            # Skip redundant GSM antenna insights
            if "GSM cellular antennas" in insight and "provide cellular coverage" in insight:
                if not gsm_insight_added:
                    gsm_count = antenna_counts.get('GSM Antenna', 0)
                    if gsm_count > 0:
                        # Add the insight with the correct count
                        filtered_insights.append(f"GSM cellular antennas ({gsm_count}) provide cellular coverage.")
                    gsm_insight_added = True
                continue
                
            # For all other insights, add them if they're not already in the list
            if insight not in filtered_insights:
                filtered_insights.append(insight)
        
        # Add tower type insight if not yet present
        tower_type_insight_present = False
        for insight in filtered_insights:
            if "tower identified" in insight:
                tower_type_insight_present = True
                break
                
        if not tower_type_insight_present and most_common_type != "Unknown" and most_common_type != "No Tower Detected":
            if most_common_type == "Lattice Tower":
                filtered_insights.insert(0, "Lattice tower identified - suitable for high load capacity with multiple antennas.")
            elif most_common_type == "Monopole Tower":
                filtered_insights.insert(0, "Monopole tower identified - typically has cleaner aesthetics but lower capacity.")
        
        # Compile results
        combined_results = {
            'tower_type': most_common_type,
            'height_estimate': avg_height,
            'antenna_counts': antenna_counts,
            'total_antennas': total_antennas,
            'insights': filtered_insights,
            'best_frame_info': best_frame_info
        }
        
        return combined_results 
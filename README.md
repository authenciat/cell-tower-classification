# Verizon Tower Detection and Analysis System

A comprehensive system for detecting, classifying, and analyzing cell towers from drone footage. This project uses computer vision techniques to identify tower structures, antennas, and other components from images and videos, providing detailed insights about tower configurations.

## Features

- **Image Analysis**: Upload and analyze individual tower images
- **Video Processing**: Process drone footage to detect and analyze towers
- **Object Detection**: Identify tower structures, antennas, and other components
- **Tower Classification**: Determine tower types (lattice, monopole, etc.)
- **Antenna Counting**: Accurately count and classify different antenna types
- **Measurement Estimation**: Estimate tower height and dimensions
- **Visualization**: Visual display of detection results with bounding boxes
- **Insight Generation**: Generate actionable insights about tower configurations

## System Requirements

- Python 3.8+
- OpenCV
- PyTorch
- Gradio
- Required model files (YOLOv8 weights)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/authenciat/cell-tower-classification.git
cd cell-tower-classification
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the web interface:
```bash
python app.py
```

2. Open your web browser and navigate to `http://127.0.0.1:7860`

3. Use the interface to:
   - Upload images or videos of cell towers
   - Adjust detection confidence levels
   - View analysis results and visualizations
   - Extract insights about tower configurations

## Project Structure

- `app.py`: Main application with Gradio web interface
- `src/`: Core modules
  - `detection.py`: Tower detection module using YOLOv8
  - `tower_analyzer.py`: Tower analysis and classification
  - `visualization.py`: Visualization utilities
  - `utils.py`: Helper functions for video processing and file operations

## Model Training

The detection models are trained using YOLOv8 on a custom dataset of cell tower images. The models can identify various tower components including:

- Tower structures (lattice, monopole)
- Antenna types (GSM, microwave, etc.)
- Tower accessories and equipment

## License

[MIT License](LICENSE)

## Acknowledgments

- YOLOv8 for object detection
- Gradio for the web interface
- Verizon for project specifications 
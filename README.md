# 🔦 SpotLight - Real-time Object Detection

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/YOLOv8-Latest-green.svg" alt="YOLOv8">
  <img src="https://img.shields.io/badge/Flask-2.0+-red.svg" alt="Flask">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
</p>

SpotLight is a modern, real-time object detection application that uses YOLOv8 to identify furniture, electronics, and everyday objects through your webcam. Built with a sleek web interface, it provides instant detection with category filtering and comprehensive statistics.

## ✨ Features

- 🎥 **Real-time Detection** - Live webcam feed with instant object detection
- 🎯 **Multi-Category Support** - Detects 30+ object types across 5 categories
- 🎨 **Modern Web Interface** - Responsive dark-themed UI with real-time updates
- 📊 **Live Statistics** - FPS counter, detection history, and category distribution
- 🔍 **Smart Filtering** - Filter detections by category (Furniture, Electronics, etc.)
- 📸 **Screenshot Capture** - Save detection results with one click
- 🚀 **Optimized Performance** - Hardware acceleration support for Apple Silicon

## 🖼️ Screenshots

<details>
<summary>View Screenshots</summary>

### Web Interface
The modern dark-themed interface provides real-time video feed with detection overlays.

### Detection Categories
- 🪑 **Furniture**: Chairs, tables, beds, couches
- 💻 **Electronics**: Monitors, laptops, keyboards, phones
- 🍴 **Kitchen**: Cups, bottles, utensils, bowls
- 📚 **Office/Decor**: Books, plants, clocks, vases
- 👤 **Living**: People, pets, bags

</details>

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Webcam/Camera
- macOS, Windows, or Linux

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/spotlight.git
cd spotlight
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Usage

#### 🌐 Web Application (Recommended)

1. Start the web server:
```bash
python run_webapp.py
```

2. Open your browser and navigate to:
```
http://localhost:8080
```

3. Click "Detect Once" or enable "Continuous Mode" to start detecting!

#### 💻 Command Line Interface

For real-time camera detection:
```bash
python run_cli.py
```

#### 📚 Examples

Check the `examples/` directory for more usage examples:
```bash
# Basic image detection
python examples/run.py

# Simple camera detection
python examples/realtime_simple.py

# YOLOv8 detection demo
python examples/realtime_yolov8.py
```

## 🛠️ Project Structure

```
spotlight/
├── run_webapp.py          # Web application launcher
├── run_cli.py            # CLI launcher
├── src/                  # Main application code
│   ├── __init__.py
│   ├── webapp.py         # Flask web application
│   └── realtime_all_items.py  # CLI detection
├── templates/            # HTML templates
│   ├── detection.html    # Main web interface
│   └── index.html        # Upload interface
├── examples/             # Example scripts
│   ├── run.py           # Basic image detection
│   ├── realtime_simple.py    # Simple camera detection
│   └── realtime_yolov8.py    # YOLOv8 detection
├── scripts/              # Utility scripts
│   ├── test_detection.py # Test script
│   ├── camera_test.py    # Camera test
│   └── check_classes.py  # Check YOLO classes
├── static/              # Static files (auto-created)
├── uploads/             # Upload directory (auto-created)
├── results/             # Results directory (auto-created)
├── requirements.txt     # Python dependencies
├── LICENSE              # MIT License
├── README.md           # This file
└── .gitignore          # Git ignore rules
```

## 📋 Available Scripts

- `webapp.py` - Full-featured web application with UI
- `realtime_all_items.py` - Terminal-based real-time detection
- `realtime_yolov8.py` - Basic YOLOv8 furniture detection
- `test_detection.py` - Test script for debugging

## ⚙️ Configuration

### Camera Settings
Modify camera resolution in any script:
```python
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
```

### Detection Confidence
Adjust the confidence threshold (0.0 to 1.0):
```python
if conf > 0.4:  # Default is 0.4
```

### Port Configuration
Change the web server port in `webapp.py`:
```python
app.run(debug=False, threaded=True, port=8080)
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) for the amazing object detection model
- [Flask](https://flask.palletsprojects.com/) for the web framework
- [OpenCV](https://opencv.org/) for computer vision capabilities

## 🐛 Troubleshooting

### Camera Permission (macOS)
If you see camera permission errors:
1. Go to System Settings → Privacy & Security → Camera
2. Enable camera access for Terminal or your Python application

### Port Already in Use
If port 8080 is busy, either:
- Stop the process using the port: `lsof -ti:8080 | xargs kill -9`
- Or change the port in `webapp.py`

### Performance Issues
- Ensure you're using a supported Python version (3.8+)
- For Apple Silicon Macs, the app automatically uses Metal Performance Shaders
- Reduce camera resolution for better performance

---

<p align="center">
  Made with ❤️ by Your Name
</p>
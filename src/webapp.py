from flask import Flask, render_template, Response, jsonify
import cv2
from ultralytics import YOLO
import torch
import json
import threading
import time
from datetime import datetime

app = Flask(__name__)

# Global variables
camera = None
model = None
detection_enabled = False
continuous_mode = False
filter_category = None
last_detections = []
stats = {
    'total_detections': 0,
    'detection_history': [],
    'fps': 0
}

# Define items to detect (same as before)
items_to_detect = {
    # Furniture
    56: 'chair', 57: 'couch', 59: 'bed', 60: 'dining table',
    # Electronics
    62: 'tv/monitor', 63: 'laptop', 64: 'mouse', 65: 'remote', 
    66: 'keyboard', 67: 'cell phone',
    # Kitchen items
    46: 'banana', 47: 'apple', 49: 'orange', 43: 'knife',
    44: 'spoon', 45: 'fork', 39: 'bottle', 41: 'cup',
    40: 'wine glass', 42: 'bowl',
    # Office/Room items
    73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 84: 'potted plant',
    # Other
    0: 'person', 15: 'cat', 16: 'dog', 26: 'handbag', 
    27: 'suitcase', 28: 'backpack',
}

categories = {
    'Furniture': [56, 57, 59, 60],
    'Electronics': [62, 63, 64, 65, 66, 67],
    'Kitchen': [39, 40, 41, 42, 43, 44, 45, 46, 47, 49],
    'Office/Decor': [73, 74, 75, 76, 84],
    'Living': [0, 15, 16, 26, 27, 28]
}

category_colors = {
    'Furniture': '#4CAF50',      # Green
    'Electronics': '#2196F3',     # Blue
    'Kitchen': '#FFEB3B',         # Yellow
    'Office/Decor': '#9C27B0',    # Purple
    'Living': '#FF9800'           # Orange
}

def get_category(class_id):
    for cat, items in categories.items():
        if class_id in items:
            return cat
    return 'Other'

def init_camera():
    global camera
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    camera.set(cv2.CAP_PROP_FPS, 30)

def init_model():
    global model
    print("Loading YOLOv8 model...")
    model = YOLO('yolov8m.pt')
    print("Model loaded!")

def generate_frames():
    global camera, detection_enabled, continuous_mode, filter_category, last_detections, stats
    
    fps_time = time.time()
    fps_counter = 0
    
    while True:
        if camera is None:
            continue
            
        success, frame = camera.read()
        if not success:
            break
        
        # Calculate FPS
        fps_counter += 1
        if time.time() - fps_time > 1:
            stats['fps'] = fps_counter / (time.time() - fps_time)
            fps_counter = 0
            fps_time = time.time()
        
        # Run detection if enabled
        if detection_enabled and (continuous_mode or detection_enabled):
            try:
                with torch.no_grad():
                    results = model(frame, verbose=False)
                
                detected_items = []
                
                if len(results[0].boxes) > 0:
                    for box in results[0].boxes:
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        
                        if cls_id in items_to_detect and conf > 0.4:
                            category = get_category(cls_id)
                            
                            # Apply filter
                            if filter_category and category != filter_category:
                                continue
                            
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            
                            detected_items.append({
                                'name': items_to_detect[cls_id],
                                'category': category,
                                'confidence': round(conf, 2),
                                'bbox': [x1, y1, x2, y2],
                                'color': category_colors.get(category, '#FFFFFF')
                            })
                
                last_detections = detected_items
                
                # Update stats
                if detected_items:
                    stats['total_detections'] += len(detected_items)
                    stats['detection_history'].append({
                        'timestamp': datetime.now().strftime('%H:%M:%S'),
                        'count': len(detected_items),
                        'items': [item['name'] for item in detected_items[:5]]  # First 5 items
                    })
                    # Keep only last 10 entries
                    stats['detection_history'] = stats['detection_history'][-10:]
                
                # Draw on frame
                for item in detected_items:
                    x1, y1, x2, y2 = item['bbox']
                    # Convert hex color to BGR
                    color = tuple(int(item['color'][i:i+2], 16) for i in (5, 3, 1))
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    label = f"{item['name']}: {item['confidence']}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    cv2.rectangle(frame, (x1, y1-20), (x1+label_size[0], y1), color, -1)
                    cv2.putText(frame, label, (x1, y1-5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Reset single detection
                if not continuous_mode:
                    detection_enabled = False
                    
            except Exception as e:
                print(f"Detection error: {e}")
        
        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('detection.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detect', methods=['POST'])
def detect():
    global detection_enabled
    detection_enabled = True
    return jsonify({'status': 'detection_triggered'})

@app.route('/toggle_continuous', methods=['POST'])
def toggle_continuous():
    global continuous_mode, detection_enabled
    continuous_mode = not continuous_mode
    detection_enabled = continuous_mode
    return jsonify({'continuous': continuous_mode})

@app.route('/set_filter/<category>', methods=['POST'])
def set_filter(category):
    global filter_category
    if category == 'all':
        filter_category = None
    else:
        filter_category = category
    return jsonify({'filter': filter_category})

@app.route('/get_detections')
def get_detections():
    global last_detections, stats
    
    # Count by category
    category_counts = {}
    for item in last_detections:
        cat = item['category']
        category_counts[cat] = category_counts.get(cat, 0) + 1
    
    return jsonify({
        'detections': last_detections,
        'category_counts': category_counts,
        'stats': stats,
        'timestamp': datetime.now().strftime('%H:%M:%S')
    })

@app.route('/save_screenshot', methods=['POST'])
def save_screenshot():
    if camera is None:
        return jsonify({'error': 'Camera not initialized'}), 400
    
    success, frame = camera.read()
    if success:
        filename = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(filename, frame)
        return jsonify({'filename': filename})
    
    return jsonify({'error': 'Failed to capture frame'}), 500

if __name__ == '__main__':
    print("Initializing camera and model...")
    init_camera()
    init_model()
    print("Starting web server...")
    print("Open http://localhost:8080 in your browser")
    app.run(debug=False, threaded=True, port=8080)
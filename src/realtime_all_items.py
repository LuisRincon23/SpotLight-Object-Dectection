import cv2
from ultralytics import YOLO
import time
import torch

print("Multi-Object Detection with YOLOv8")
print("=" * 40)

# Load YOLOv8 model
print("Loading YOLOv8 model...")
model = YOLO('yolov8m.pt')

# Define categories of items to detect
# YOLOv8 COCO classes we're interested in:
items_to_detect = {
    # Furniture
    56: 'chair',
    57: 'couch',
    59: 'bed',
    60: 'dining table',
    
    # Electronics
    62: 'tv/monitor',
    63: 'laptop',
    64: 'mouse',
    65: 'remote',
    66: 'keyboard',
    67: 'cell phone',
    
    # Kitchen items
    46: 'banana',
    47: 'apple',
    49: 'orange',
    43: 'knife',
    44: 'spoon',
    45: 'fork',
    39: 'bottle',
    41: 'cup',
    40: 'wine glass',
    42: 'bowl',
    
    # Office/Room items
    73: 'book',
    74: 'clock',
    75: 'vase',
    76: 'scissors',
    84: 'potted plant',
    
    # Other useful items
    0: 'person',
    15: 'cat',
    16: 'dog',
    26: 'handbag',
    27: 'suitcase',
    28: 'backpack',
}

# Group items by category for display
categories = {
    'Furniture': [56, 57, 59, 60],
    'Electronics': [62, 63, 64, 65, 66, 67],
    'Kitchen': [39, 40, 41, 42, 43, 44, 45, 46, 47, 49],
    'Office/Decor': [73, 74, 75, 76, 84],
    'Living': [0, 15, 16, 26, 27, 28]
}

# Colors for each category
category_colors = {
    'Furniture': (0, 255, 0),      # Green
    'Electronics': (255, 0, 0),     # Blue
    'Kitchen': (0, 255, 255),       # Yellow
    'Office/Decor': (255, 0, 255),  # Magenta
    'Living': (0, 165, 255)         # Orange
}

print("Model loaded!")
print(f"\nDetecting {len(items_to_detect)} types of objects:")
for cat, items in categories.items():
    print(f"\n{cat}:")
    for item_id in items:
        if item_id in items_to_detect:
            print(f"  - {items_to_detect[item_id]}")

# Open camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("\nError: Cannot open camera")
    exit()

print("\n✅ Camera ready!")
print("\nCONTROLS:")
print("- SPACE: Detect objects")
print("- 'c': Toggle continuous mode")
print("- 'f': Filter mode (show only specific categories)")
print("- 's': Save screenshot")
print("- 'q': Quit")

cv2.namedWindow('Object Detection', cv2.WINDOW_NORMAL)

continuous_mode = False
filter_mode = None
last_results = None
fps_time = time.time()
fps_counter = 0
current_fps = 0

def get_category(class_id):
    """Get category name for a class ID"""
    for cat, items in categories.items():
        if class_id in items:
            return cat
    return 'Other'

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    
    display_frame = frame.copy()
    
    # Calculate FPS
    fps_counter += 1
    if time.time() - fps_time > 1:
        current_fps = fps_counter / (time.time() - fps_time)
        fps_counter = 0
        fps_time = time.time()
    
    # Show status
    status_y = 30
    cv2.putText(display_frame, f"FPS: {current_fps:.1f}", (10, status_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    status_y += 25
    
    mode_text = "CONTINUOUS" if continuous_mode else "Press SPACE"
    color = (0, 255, 0) if continuous_mode else (255, 255, 0)
    cv2.putText(display_frame, f"Mode: {mode_text}", (10, status_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    status_y += 25
    
    if filter_mode:
        cv2.putText(display_frame, f"Filter: {filter_mode}", (10, status_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    
    # Process frame
    key = cv2.waitKey(1) & 0xFF
    
    if continuous_mode or key == 32:  # SPACE
        try:
            # Run detection
            with torch.no_grad():
                results = model(frame, verbose=False)
            
            # Process results
            detected_items = []
            
            if len(results[0].boxes) > 0:
                for box in results[0].boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    # Check if we want to detect this item
                    if cls_id in items_to_detect and conf > 0.4:
                        category = get_category(cls_id)
                        
                        # Apply filter if active
                        if filter_mode and category != filter_mode:
                            continue
                        
                        detected_items.append({
                            'name': items_to_detect[cls_id],
                            'category': category,
                            'conf': conf,
                            'box': box
                        })
                
                if detected_items:
                    if not continuous_mode or key == 32:
                        print(f"\n✅ Found {len(detected_items)} items:")
                        # Group by category
                        by_category = {}
                        for item in detected_items:
                            cat = item['category']
                            if cat not in by_category:
                                by_category[cat] = []
                            by_category[cat].append(item)
                        
                        for cat, items in by_category.items():
                            print(f"\n{cat}:")
                            for item in items:
                                print(f"  - {item['name']}: {item['conf']:.2f}")
                    
                    last_results = detected_items
                else:
                    if not continuous_mode:
                        print("\n❌ No items detected")
                        
        except Exception as e:
            print(f"\n⚠️ Error: {str(e)}")
    
    # Draw detections
    if last_results:
        # Count items by category
        category_counts = {}
        
        for item in last_results:
            box = item['box']
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Get color for category
            color = category_colors.get(item['category'], (255, 255, 255))
            
            # Draw box
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{item['name']}: {item['conf']:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(display_frame, (x1, y1-20), (x1+label_size[0], y1), color, -1)
            cv2.putText(display_frame, label, (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Count for summary
            cat = item['category']
            category_counts[cat] = category_counts.get(cat, 0) + 1
        
        # Show summary
        summary_y = display_frame.shape[0] - 10
        for cat, count in sorted(category_counts.items()):
            color = category_colors.get(cat, (255, 255, 255))
            text = f"{cat}: {count}"
            cv2.putText(display_frame, text, (10, summary_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            summary_y -= 20
    
    # Show controls
    cv2.putText(display_frame, "c:continuous f:filter s:save q:quit", 
                (10, display_frame.shape[0] - 120), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    cv2.imshow('Object Detection', display_frame)
    
    # Handle keys
    if key == ord('q'):
        break
    elif key == ord('c'):
        continuous_mode = not continuous_mode
        print(f"\n{'🟢' if continuous_mode else '🔴'} Continuous: {'ON' if continuous_mode else 'OFF'}")
        if not continuous_mode:
            last_results = None
    elif key == ord('f'):
        # Cycle through filters
        if filter_mode is None:
            filter_mode = 'Furniture'
        elif filter_mode == 'Furniture':
            filter_mode = 'Electronics'
        elif filter_mode == 'Electronics':
            filter_mode = 'Kitchen'
        elif filter_mode == 'Kitchen':
            filter_mode = 'Office/Decor'
        elif filter_mode == 'Office/Decor':
            filter_mode = 'Living'
        else:
            filter_mode = None
        
        print(f"\n🔍 Filter: {filter_mode if filter_mode else 'OFF'}")
        last_results = None
    elif key == ord('s'):
        filename = f"detection_{time.strftime('%H%M%S')}.jpg"
        cv2.imwrite(filename, display_frame)
        print(f"\n📸 Saved: {filename}")

cap.release()
cv2.destroyAllWindows()
print("\n✅ Done!")
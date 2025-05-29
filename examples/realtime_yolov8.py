import cv2
from ultralytics import YOLO
import time
import torch

print("Furniture Detection with YOLOv8")
print("=" * 40)

# Download and use standard YOLOv8 model
print("Loading YOLOv8 model...")
model = YOLO('yolov8m.pt')  # Medium model for good balance

# Standard COCO furniture classes in YOLOv8
# chair: 56, couch: 57, bed: 59, dining table: 60, toilet: 61
furniture_classes = {56: 'chair', 57: 'couch', 59: 'bed', 60: 'dining table'}

print("Model loaded!")
print(f"Looking for: {list(furniture_classes.values())}")

# Open camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Error: Cannot open camera")
    exit()

print("\n✅ Camera ready!")
print("\nINSTRUCTIONS:")
print("- Point camera at furniture (chairs, couches, beds, tables)")
print("- Press SPACE to detect")
print("- Press 'c' for continuous mode")
print("- Press 'q' to quit")

cv2.namedWindow('YOLOv8 Furniture Detection', cv2.WINDOW_NORMAL)

continuous_mode = False
last_results = None

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    
    display_frame = frame.copy()
    
    # Show mode
    mode_text = "CONTINUOUS MODE" if continuous_mode else "Press SPACE to detect"
    color = (0, 255, 0) if continuous_mode else (255, 255, 0)
    cv2.putText(display_frame, mode_text, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    # Process frame if in continuous mode or space pressed
    key = cv2.waitKey(1) & 0xFF
    
    if continuous_mode or key == 32:  # SPACE
        try:
            # Run detection
            with torch.no_grad():
                results = model(frame, verbose=False)
            
            # Process results
            furniture_found = []
            
            if len(results[0].boxes) > 0:
                for box in results[0].boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    # Check if it's furniture
                    if cls_id in furniture_classes and conf > 0.4:
                        furniture_found.append({
                            'name': furniture_classes[cls_id],
                            'conf': conf,
                            'box': box
                        })
                
                if furniture_found:
                    print(f"\n✅ Found {len(furniture_found)} furniture items:")
                    for item in furniture_found:
                        print(f"   - {item['name']}: {item['conf']:.2f}")
                    last_results = furniture_found
                else:
                    if not continuous_mode:
                        print("\n❌ No furniture detected")
            else:
                if not continuous_mode:
                    print("\n❌ No objects detected")
                    
        except Exception as e:
            print(f"\n⚠️ Error: {str(e)}")
    
    # Draw detections
    if last_results:
        for item in last_results:
            box = item['box']
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Draw box
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"{item['name']}: {item['conf']:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(display_frame, (x1, y1-25), (x1+label_size[0], y1), (0, 255, 0), -1)
            cv2.putText(display_frame, label, (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Show FPS
    cv2.putText(display_frame, "Press 'c' for continuous mode", (10, 460), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    cv2.imshow('YOLOv8 Furniture Detection', display_frame)
    
    # Handle keys
    if key == ord('q'):
        break
    elif key == ord('c'):
        continuous_mode = not continuous_mode
        print(f"\nContinuous mode: {'ON' if continuous_mode else 'OFF'}")
        if not continuous_mode:
            last_results = None
    elif key == ord('s'):
        filename = f"yolov8_furniture_{time.strftime('%H%M%S')}.jpg"
        cv2.imwrite(filename, display_frame)
        print(f"\n📸 Saved: {filename}")

cap.release()
cv2.destroyAllWindows()
print("\n✅ Done!")
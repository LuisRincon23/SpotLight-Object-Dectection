import cv2
from ultralytics import YOLO
import time

print("Simple Furniture Detection - Real Time")
print("=" * 40)

# Load model
print("Loading model...")
model = YOLO("yoloe-11l-seg.pt")

# Set furniture categories
furniture_names = ["furniture", "chair", "table", "sofa", "couch", "desk", "bed", 
                  "cabinet", "shelf", "dresser", "bench", "stool", "ottoman"]
model.set_classes(furniture_names, model.get_text_pe(furniture_names))

# Open camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Error: Cannot open camera")
    exit()

print("\n✅ Camera ready!")
print("\nINSTRUCTIONS:")
print("1. Point camera at furniture (chairs, tables, desks, etc.)")
print("2. Press 'SPACE' to detect furniture in current frame")
print("3. Press 'c' for continuous detection mode")
print("4. Press 's' to save screenshot")
print("5. Press 'q' to quit")
print("\nWindow will show camera feed...")

# Create window
cv2.namedWindow('Camera Feed', cv2.WINDOW_NORMAL)

continuous_mode = False
last_results = None

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    
    display_frame = frame.copy()
    
    # Show mode
    mode_text = "Mode: Continuous" if continuous_mode else "Mode: Manual (press SPACE)"
    cv2.putText(display_frame, mode_text, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    # In continuous mode or when space is pressed
    if continuous_mode or cv2.waitKey(1) == 32:  # 32 is SPACE
        print("\n🔍 Detecting furniture...")
        
        # Run detection
        start = time.time()
        results = model.predict(frame, conf=0.3, verbose=False)
        inference_time = (time.time() - start) * 1000
        
        last_results = results
        
        # Print results
        if len(results[0].boxes) > 0:
            print(f"✅ Found {len(results[0].boxes)} furniture items in {inference_time:.0f}ms:")
            for box in results[0].boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                print(f"   - {furniture_names[cls]}: {conf:.2f}")
        else:
            print(f"❌ No furniture detected in {inference_time:.0f}ms")
            print("   Try: Better lighting, different angle, or closer distance")
    
    # Draw last results if available
    if last_results is not None and len(last_results[0].boxes) > 0:
        for box in last_results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            
            # Draw box
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"{furniture_names[cls]}: {conf:.2f}"
            cv2.putText(display_frame, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Show frame
    cv2.imshow('Camera Feed', display_frame)
    
    # Handle keys
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        continuous_mode = not continuous_mode
        print(f"\n{'🟢' if continuous_mode else '🔴'} Continuous mode: {'ON' if continuous_mode else 'OFF'}")
    elif key == ord('s'):
        filename = f"furniture_capture_{time.strftime('%H%M%S')}.jpg"
        cv2.imwrite(filename, display_frame)
        print(f"\n📸 Saved: {filename}")

cap.release()
cv2.destroyAllWindows()
print("\n👋 Goodbye!")
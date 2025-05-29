import cv2
import numpy as np
from ultralytics import YOLO
import glob
import os

print("=== Furniture Detection Test ===")

# Initialize YOLO model
print("Loading YOLO model...")
model = YOLO("yoloe-11l-seg.pt")

# Set furniture categories
furniture_names = ["furniture", "chair", "table", "sofa", "couch", "desk", "bed", 
                  "cabinet", "shelf", "dresser", "bench", "stool", "ottoman"]
model.set_classes(furniture_names, model.get_text_pe(furniture_names))

# Test 1: Original screenshot
print("\n1. Testing with original screenshot...")
png_files = glob.glob("Screenshot*.png")
if png_files:
    image_path = png_files[0]
    print(f"   Using: {image_path}")
    
    # Load and display image info
    img = cv2.imread(image_path)
    print(f"   Image shape: {img.shape}")
    
    # Run detection
    results = model.predict(image_path, conf=0.25, verbose=True)
    
    if len(results[0].boxes) > 0:
        print(f"   ✅ Detected {len(results[0].boxes)} furniture items!")
        for i, box in enumerate(results[0].boxes):
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            print(f"      - {furniture_names[cls]}: {conf:.2f}")
    else:
        print("   ❌ No furniture detected")
    
    # Save result
    results[0].save(filename="test_screenshot_result.jpg")
    print("   Saved: test_screenshot_result.jpg")

# Test 2: Capture from camera
print("\n2. Testing with camera capture...")
cap = cv2.VideoCapture(0)
if cap.isOpened():
    print("   Camera opened, capturing frame...")
    ret, frame = cap.read()
    if ret:
        # Save the captured frame
        cv2.imwrite("test_camera_frame.jpg", frame)
        print(f"   Captured frame shape: {frame.shape}")
        
        # Run detection
        results = model.predict(frame, conf=0.25, verbose=True)
        
        if len(results[0].boxes) > 0:
            print(f"   ✅ Detected {len(results[0].boxes)} furniture items!")
            for i, box in enumerate(results[0].boxes):
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                print(f"      - {furniture_names[cls]}: {conf:.2f}")
        else:
            print("   ❌ No furniture detected")
        
        # Save annotated result
        annotated = results[0].plot()
        cv2.imwrite("test_camera_result.jpg", annotated)
        print("   Saved: test_camera_result.jpg")
    cap.release()
else:
    print("   ❌ Could not open camera")

# Test 3: Lower confidence threshold
print("\n3. Testing with lower confidence (0.1)...")
if 'frame' in locals():
    results = model.predict(frame, conf=0.1, verbose=False)
    if len(results[0].boxes) > 0:
        print(f"   ✅ With lower confidence, detected {len(results[0].boxes)} items")
    else:
        print("   ❌ Still no detections even with low confidence")

# Test 4: Test the model with basic prediction
print("\n4. Testing model directly...")
test_img = np.zeros((640, 640, 3), dtype=np.uint8)
test_img[:] = (128, 128, 128)  # Gray image
try:
    results = model.predict(test_img, verbose=False)
    print("   ✅ Model inference working correctly")
except Exception as e:
    print(f"   ❌ Model error: {e}")

print("\n=== Test Summary ===")
print("Check the saved images:")
print("- test_screenshot_result.jpg (annotated screenshot)")
print("- test_camera_frame.jpg (raw camera capture)")
print("- test_camera_result.jpg (annotated camera capture)")
print("\nIf no furniture was detected in camera, try:")
print("1. Point camera at actual furniture")
print("2. Ensure good lighting")
print("3. Try different angles/distances")
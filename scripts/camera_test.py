import cv2
import os

print("Testing camera access...")
print(f"Running as user: {os.getenv('USER')}")

# Try to open camera
cap = cv2.VideoCapture(0)

if cap.isOpened():
    print("✅ Camera opened successfully!")
    ret, frame = cap.read()
    if ret:
        print("✅ Successfully captured a frame")
        print(f"Frame size: {frame.shape}")
    cap.release()
else:
    print("❌ Failed to open camera")
    print("\nTo fix this on Mac:")
    print("1. Open System Settings")
    print("2. Go to Privacy & Security > Camera")
    print("3. Find 'Terminal' or 'Python' in the list")
    print("4. Toggle the switch to enable camera access")
    print("5. You may need to restart Terminal after granting permission")
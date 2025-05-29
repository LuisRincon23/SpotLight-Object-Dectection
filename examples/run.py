from ultralytics import YOLO
import glob
import os

# Initialize YOLOE model
model = YOLO("yoloe-11l-seg.pt")

# Set text prompt to detect furniture
names = ["furniture", "chair", "table", "sofa", "couch", "desk", "bed", "cabinet", "shelf", "dresser", "bench", "stool", "ottoman"]
model.set_classes(names, model.get_text_pe(names))

# Find the screenshot file
png_files = glob.glob("Screenshot*.png")
if not png_files:
    raise FileNotFoundError("No screenshot PNG file found")
image_path = png_files[0]
print(f"Processing image: {image_path}")

# Run detection on the image
results = model.predict(image_path)

# Show results
results[0].show()

# Save annotated image
results[0].save(filename="furniture_detection_result.jpg")
print("Detection complete! Results saved as 'furniture_detection_result.jpg'")
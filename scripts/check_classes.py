from ultralytics import YOLO

# Load model
model = YOLO("yoloe-11l-seg.pt")

print("Available classes in YOLO model:")
print("=" * 40)

# Print all classes
furniture_classes = []
for i, name in model.names.items():
    print(f"{i}: {name}")
    
    # Check if it's furniture
    if any(furn in name.lower() for furn in ['chair', 'couch', 'sofa', 'bed', 'desk', 
                                              'table', 'bench', 'cabinet', 'shelf', 
                                              'dresser', 'stool', 'ottoman', 'furniture',
                                              'dining table', 'diningtable']):
        furniture_classes.append((i, name))

print("\n\nFurniture-related classes:")
print("=" * 40)
for i, name in furniture_classes:
    print(f"{i}: {name}")
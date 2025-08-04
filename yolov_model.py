import os
import gdown
from ultralytics import YOLO
import cv2

MODEL_URL = "https://drive.google.com/uc?id=1Gps_dqoQkJMIMYyB7Aox1N07SXtnQr9o"
MODEL_PATH = "best12.pt"

# ✅ Do not load model globally
model = None

def get_model():
    """Load YOLO model only once (lazy loading)."""
    global model
    if model is None:
        if not os.path.exists(MODEL_PATH):
            print("⬇️ Downloading YOLO model from Google Drive...")
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
        print("✅ Loading YOLO model into memory...")
        model = YOLO(MODEL_PATH)
    return model

def run_yolo(input_path, output_path):
    """Run YOLO model on input image and save annotated result."""
    yolo_model = get_model()
    results = yolo_model(input_path, save=False)
    img = results[0].plot()  # Annotated image with bounding boxes
    cv2.imwrite(output_path, img)
    print(f"✅ Processed image saved to: {output_path}")

import os
import gdown
from ultralytics import YOLO
import cv2

MODEL_URL = "https://drive.google.com/uc?id=1Gps_dqoQkJMIMYyB7Aox1N07SXtnQr9o"
MODEL_PATH = "best12.pt"

if not os.path.exists(MODEL_PATH):
    print("⬇️ Downloading YOLO model from Google Drive...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# ✅ Load model once
model = YOLO(MODEL_PATH)

def run_yolo(input_path, output_path):
    """Run YOLO model on input image and save annotated result."""
    results = model(input_path, save=False)
    img = results[0].plot()  # Annotated image with bounding boxes
    cv2.imwrite(output_path, img)
    print(f"✅ Processed image saved to: {output_path}")

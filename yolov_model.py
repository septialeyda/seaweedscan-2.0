import os
import requests
from ultralytics import YOLO
import gdown

MODEL_PATH = "best12.pt"
MODEL_URL = "https://drive.google.com/uc?id=1Gps_dqoQkJMIMYyB7Aox1N07SXtnQr9o"

# Download model if not exists
if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# Initialize model
model = YOLO(MODEL_PATH)

# ✅ Add this function
def run_yolo(image_path):
    """
    Runs YOLO inference on the given image.
    :param image_path: Path to input image
    :return: YOLO prediction results
    """
    results = model.predict(image_path, conf=0.25)
    return results

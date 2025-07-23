import os
import gdown
from ultralytics import YOLO

MODEL_PATH = "best12.pt"
MODEL_URL = "https://drive.google.com/uc?id=1Gps_dqoQkJMIMYyB7Aox1N07SXtnQr9o"

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("⬇️ Downloading YOLO model from Google Drive...")
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

download_model()
model = YOLO(MODEL_PATH)

def run_yolo(input_path, output_path):
    results = model(input_path, save=False)
    img = results[0].plot()
    cv2.imwrite(output_path, img)

import os
import requests
from ultralytics import YOLO

MODEL_URL = "https://huggingface.co/septialeyda/seaweed-yolov8-model/resolve/main/best-12.pt"
MODEL_PATH = "best-12.pt"

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("⬇️ Downloading YOLO model from Hugging Face...")
        r = requests.get(MODEL_URL, stream=True)
        with open(MODEL_PATH, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

download_model()
model = YOLO(MODEL_PATH)

import os
import requests
from ultralytics import YOLO
import gdown

MODEL_PATH = "best12.pt"
MODEL_URL = "https://drive.google.com/uc?id=1Gps_dqoQkJMIMYyB7Aox1N07SXtnQr9o"

if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)


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

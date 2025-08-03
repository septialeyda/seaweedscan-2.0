from ultralytics import YOLO
import cv2

# Load model once
model = YOLO('best12.pt')  # or your own .pt model

def run_yolo(input_path, output_path):
    results = model(input_path, save=False)
    img = results[0].plot()  # returns image array with bounding boxes
    cv2.imwrite(output_path, img)

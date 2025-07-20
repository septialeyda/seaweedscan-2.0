from flask import Flask, render_template, request, redirect, url_for
import os
from yolov_model import run_yolo

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
OUTPUT_FOLDER = 'static/outputs'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(image_path)

            output_path = os.path.join(app.config['OUTPUT_FOLDER'], file.filename)
            run_yolo(image_path, output_path)

            return render_template('index.html', uploaded=True,
                                   input_img=image_path, output_img=output_path)
    return render_template('index.html', uploaded=False)

if __name__ == '__main__':
    app.run(debug=True)

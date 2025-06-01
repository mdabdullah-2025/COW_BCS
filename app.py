import os
from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
import base64
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Disable fusion and set temp config directory to prevent permission issues
os.environ['YOLO_DO_NOT_FUSE'] = '1'
os.environ['YOLO_CONFIG_DIR'] = '/tmp'

app = Flask(__name__)

# Global model variable
model = None

def load_model():
    """Load the YOLO model with error handling"""
    global model
    try:
        logger.info("Loading YOLO model...")
        model = YOLO("best6.pt")
        logger.info("Model loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return False

# Load model when starting the app
load_model()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template("index.html")

    # Handle POST request
    if model is None:
        return render_template("index.html", error="Model not loaded")

    if "image" not in request.files:
        return render_template("index.html", error="No image uploaded")

    file = request.files["image"]
    if file.filename == "":
        return render_template("index.html", error="No image selected")

    try:
        # Read and process image
        img_bytes = file.read()
        img_np = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

        if img is None:
            return render_template("index.html", error="Invalid image file")

        # Run prediction
        results = model.predict(source=img, save=False, show=False, verbose=False)

        if not results or len(results) == 0:
            return render_template("index.html", error="No results from model")

        original_img = results[0].orig_img.copy()
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

        boxes = results[0].boxes
        label = None
        message = None

        if boxes is not None and len(boxes) > 0:
            confs = boxes.conf.cpu().numpy()
            coords = boxes.xyxy.cpu().numpy()
            class_ids = boxes.cls.cpu().numpy()
            names = results[0].names

            best_idx = confs.argmax()
            best_box = coords[best_idx]
            best_class_id = int(class_ids[best_idx])
            best_conf = confs[best_idx]

            x1, y1, x2, y2 = map(int, best_box)
            label = f"{names[best_class_id]}"

            # Draw bounding box and label
            cv2.rectangle(original_img, (x1, y1), (x2, y2), (0, 255, 0), 4)
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
            cv2.rectangle(original_img, (x1, y1 - text_height - 15), (x1 + text_width + 10, y1), (0, 255, 0), -1)
            cv2.putText(original_img, label, (x1 + 5, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        else:
            label = "No Detection"
            message = "No object detected. Please provide a clearer image."

            text = "No Detection"
            font_scale = 2
            thickness = 4
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            x = (original_img.shape[1] - text_width) // 2
            y = (original_img.shape[0] + text_height) // 2
            cv2.putText(original_img, text, (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), thickness)

        # Convert image to base64 for displaying in HTML
        _, img_encoded = cv2.imencode('.jpg', cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR))
        image_data = base64.b64encode(img_encoded).decode('utf-8')

    except Exception as e:
        logger.error(f"Error processing image: {e}")
        return render_template("index.html", error=f"Error processing image: {str(e)}")

    return render_template("index.html", image_data=image_data, label=label, message=message)

@app.route('/predict_realtime', methods=['POST'])
def predict_realtime():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400

    try:
        # Process image
        img_bytes = file.read()
        img_np = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({'error': 'Invalid image data'}), 400

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Run prediction
        results = model.predict(source=img, save=False, show=False, verbose=False)

        boxes = results[0].boxes
        label = None
        has_detection = False

        if boxes is not None and len(boxes) > 0:
            confs = boxes.conf.cpu().numpy()
            coords = boxes.xyxy.cpu().numpy()
            class_ids = boxes.cls.cpu().numpy()
            names = results[0].names

            best_idx = confs.argmax()
            best_box = coords[best_idx]
            best_class_id = int(class_ids[best_idx])
            best_conf = confs[best_idx]

            x1, y1, x2, y2 = map(int, best_box)
            label = f"{names[best_class_id]}"
            has_detection = True

            # Draw bounding box and label
            cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 255, 0), 4)
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
            cv2.rectangle(img_rgb, (x1, y1 - text_height - 15), (x1 + text_width + 10, y1), (0, 255, 0), -1)
            cv2.putText(img_rgb, label, (x1 + 5, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

        # Convert back to JPEG
        _, img_encoded = cv2.imencode('.jpg', cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
        img_base64 = base64.b64encode(img_encoded).decode('utf-8')

        return jsonify({
            'image': img_base64,
            'label': label if has_detection else "No detection",
            'has_detection': has_detection
        })

    except Exception as e:
        logger.error(f"Error in realtime prediction: {e}")
        return jsonify({'error': f'Error processing image: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)

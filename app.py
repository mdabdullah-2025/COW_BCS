import os
from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
import base64
import logging
from io import BytesIO

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Memory optimization settings
os.environ['YOLO_DO_NOT_FUSE'] = '1'  # Disable memory-intensive fusion
os.environ['YOLO_CONFIG_DIR'] = '/tmp'  # Set config directory

app = Flask(__name__)

# Load model with reduced memory footprint
def load_model():
    try:
        logger.info("Loading optimized YOLO model...")
        model = YOLO("best6.pt")
        
        # Additional optimization settings
        model.overrides['verbose'] = False
        model.overrides['imgsz'] = 640  # Reduced resolution
        model.overrides['batch'] = 1     # Single image processing
        
        logger.info("Model loaded with optimizations")
        return model
    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        return None

model = load_model()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template("index.html")
    
    if model is None:
        return render_template("index.html", error="Model not available")

    try:
        if 'image' not in request.files:
            return render_template("index.html", error="No image uploaded")
        
        file = request.files['image']
        if file.filename == '':
            return render_template("index.html", error="No image selected")

        # Process image with memory optimization
        img_stream = BytesIO(file.read())
        img_array = np.frombuffer(img_stream.getbuffer(), np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if img is None:
            return render_template("index.html", error="Invalid image")

        # Run prediction with memory limits
        results = model.predict(
            source=img,
            imgsz=640,
            conf=0.5,  # Higher confidence threshold to reduce processing
            max_det=1,  # Limit to 1 detection
            device='cpu',  # Force CPU even if GPU is available
            verbose=False
        )

        # Process results
        if not results or len(results[0].boxes) == 0:
            output_img = cv2.putText(
                img.copy(),
                "No Detection",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )
            label = "No Detection"
        else:
            # Get best detection
            boxes = results[0].boxes
            best_idx = boxes.conf.argmax()
            box = boxes.xyxy[best_idx].cpu().numpy()
            label = results[0].names[int(boxes.cls[best_idx])]
            
            # Draw bounding box
            output_img = results[0].plot(conf=False, labels=False)
            output_img = cv2.putText(
                output_img,
                label,
                (int(box[0]), int(box[1])-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )

        # Convert to base64 with optimized compression
        _, img_encoded = cv2.imencode('.jpg', output_img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        image_data = base64.b64encode(img_encoded).decode('utf-8')

        return render_template("index.html", image_data=image_data, label=label)

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return render_template("index.html", error="Processing error")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)

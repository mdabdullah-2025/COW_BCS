from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
import base64
import os

app = Flask(__name__)

# Load the YOLO model from the ONNX file
# Ensure 'best6.onnx' is in the same directory as this Flask app,
# or provide the full path to the file.
try:
    # Explicitly set task='detect' to prevent the warning about guessing task
    model = YOLO("best6.onnx", task='detect')
    print("Model loaded successfully from best6.onnx")
except Exception as e:
    print(f"Error loading model: {e}")
    # In a production environment, you might want to log this more robustly
    # or exit the application if the model is critical.
    model = None # Set model to None if loading fails

@app.route('/', methods=['GET', 'POST'])
def index():
    """
    Handles the main page, allowing users to upload an image for cow detection.
    Processes the uploaded image, runs inference, and displays results.
    """
    if request.method == 'GET':
        return render_template("index.html")

    # Handle POST request for image upload
    image_data = None
    label = None
    message = None
    error = None

    if "image" not in request.files:
        return render_template("index.html", error="No image uploaded")

    file = request.files["image"]
    if file.filename == "":
        return render_template("index.html", error="No image selected")

    if model is None:
        # If model loading failed at startup, return an error
        return render_template("index.html", error="Model failed to load on server. Please check server logs.")

    try:
        # Read image directly into memory
        img_bytes = file.read()
        img_np = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

        if img is None:
            return render_template("index.html", error="Invalid image file. Could not decode.")

        # Run prediction on the image
        # The 'source' argument can directly take a numpy array
        results = model.predict(source=img, save=False, show=False)

        if not results or len(results) == 0:
            # This case should ideally be caught by 'boxes is not None and len(boxes) > 0' below
            # but serves as a safeguard for unexpected empty results.
            raise ValueError("No results from model prediction. This might indicate an issue with the model or input.")

        # Get the original image from results and convert to RGB for consistent drawing
        original_img = results[0].orig_img.copy()
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            # Extract confidence scores, bounding box coordinates, and class IDs
            confs = boxes.conf.cpu().numpy()
            coords = boxes.xyxy.cpu().numpy()
            class_ids = boxes.cls.cpu().numpy()
            names = results[0].names # Get class names from the model results

            # Find the detection with the highest confidence
            best_idx = confs.argmax()
            best_box = coords[best_idx]
            best_class_id = int(class_ids[best_idx])
            best_conf = confs[best_idx]

            x1, y1, x2, y2 = map(int, best_box)
            label = f"{names[best_class_id]} ({best_conf:.2f})" # Include confidence in label

            # Draw bounding box on the image
            cv2.rectangle(original_img, (x1, y1), (x2, y2), (0, 255, 0), 4) # Green rectangle
            
            # Prepare text for the label background
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
            # Draw a filled rectangle as background for the label
            cv2.rectangle(original_img, (x1, y1 - text_height - 15), (x1 + text_width + 10, y1), (0, 255, 0), -1)
            # Put the label text on the image
            cv2.putText(original_img, label, (x1 + 5, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3) # White text

        else:
            # Case where no cow is detected
            label = "No Detection"
            message = "No cow detected. Please provide a clearer image of the cow's body."

            # Add "No Cow Detected" text to the image
            text = "No Cow Detected"
            font_scale = 2
            thickness = 4
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            x = (original_img.shape[1] - text_width) // 2
            y = (original_img.shape[0] + text_height) // 2
            cv2.putText(original_img, text, (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), thickness) # Red text

            # Add a suggestion to the image
            suggestion = "Please provide a clearer image"
            font_scale_small = 0.8
            thickness_small = 2
            (sugg_width, sugg_height), _ = cv2.getTextSize(suggestion, cv2.FONT_HERSHEY_SIMPLEX, font_scale_small,
                                                           thickness_small)
            x_sugg = (original_img.shape[1] - sugg_width) // 2
            y_sugg = y + text_height + 20
            cv2.putText(original_img, suggestion, (x_sugg, y_sugg),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale_small, (0, 0, 255), thickness_small) # Red text

        # Convert the processed image (now in RGB) back to BGR for OpenCV encoding, then to base64
        _, img_encoded = cv2.imencode('.jpg', cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR))
        image_data = base64.b64encode(img_encoded).decode('utf-8')

    except Exception as e:
        error = f"Error processing image: {str(e)}"
        print(f"Error in index route: {e}") # Log the error for debugging
        return render_template("index.html", error=error)

    return render_template("index.html", image_data=image_data, label=label, message=message, error=error)


@app.route('/predict_realtime', methods=['POST'])
def predict_realtime():
    """
    Handles real-time prediction requests, typically from a webcam stream.
    Receives an image, runs inference, and returns the processed image and detection status.
    """
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400

    if model is None:
        # If model loading failed at startup, return an error
        return jsonify({'error': 'Model failed to load on server.'}), 500

    try:
        # Process image from the request
        img_bytes = file.read()
        img_np = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({'error': 'Invalid image data. Could not decode.'}), 400

        # Convert to RGB for consistency before processing
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Run prediction
        results = model.predict(source=img, save=False, show=False)

        label = "No detection" # Default label
        has_detection = False
        
        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            has_detection = True
            confs = boxes.conf.cpu().numpy()
            coords = boxes.xyxy.cpu().numpy()
            class_ids = boxes.cls.cpu().numpy()
            names = results[0].names

            best_idx = confs.argmax()
            best_box = coords[best_idx]
            best_class_id = int(class_ids[best_idx])
            best_conf = confs[best_idx]

            x1, y1, x2, y2 = map(int, best_box)
            label = f"{names[best_class_id]} ({best_conf:.2f})"

            # Draw bounding box and label on the RGB image
            cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 255, 0), 4)
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
            cv2.rectangle(img_rgb, (x1, y1 - text_height - 15), (x1 + text_width + 10, y1), (0, 255, 0), -1)
            cv2.putText(img_rgb, label, (x1 + 5, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

        # Convert the processed RGB image back to BGR for JPEG encoding
        _, img_encoded = cv2.imencode('.jpg', cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
        img_base64 = base64.b64encode(img_encoded).decode('utf-8')

        return jsonify({
            'image': img_base64,
            'label': label,
            'has_detection': has_detection
        })

    except Exception as e:
        print(f"Error in predict_realtime route: {e}") # Log the error
        return jsonify({'error': f'Error processing image: {str(e)}'}), 500


if __name__ == '__main__':
    # Ensure the 'templates' directory exists for render_template
    if not os.path.exists('templates'):
        os.makedirs('templates')
    
    # Run the Flask app
    app.run(debug=True)

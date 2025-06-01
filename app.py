from flask import Flask, render_template, request, jsonify
# from ultralytics import YOLO  # Remove this line
import cv2
import numpy as np
import base64
import onnxruntime as ort  # Add this import

app = Flask(__name__)

# Replace YOLO model loading with ONNX model loading
# model = YOLO("best6.pt")
model_path = "best6.onnx"
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if ort.get_device() == 'GPU' else ['CPUExecutionProvider']
session = ort.InferenceSession(model_path, providers=providers)

# Get model input details
model_inputs = session.get_inputs()
input_names = [input.name for input in model_inputs]
input_shape = model_inputs[0].shape
input_height, input_width = input_shape[2], input_shape[3]

# Get model output details
model_outputs = session.get_outputs()
output_names = [output.name for output in model_outputs]

# Add your class names (replace with your actual class names)
class_names = ["class1", "class2", "class3"]  # Update this with your actual class names

def preprocess_image(image):
    # Resize and normalize image for ONNX model
    image = cv2.resize(image, (input_width, input_height))
    image = image / 255.0  # Normalize to [0,1]
    image = image.transpose(2, 0, 1)  # Change from HWC to CHW
    image = np.expand_dims(image, axis=0).astype(np.float32)  # Add batch dimension
    return image

def postprocess(outputs, original_image):
    # Process the ONNX model outputs (this will vary based on your model)
    # You'll need to implement this based on how your ONNX model outputs detections
    # This is a placeholder implementation - adjust according to your model's output format
    
    # Example: assuming outputs[0] contains detections in shape [1, num_detections, 6]
    # where last dimension is [x1, y1, x2, y2, conf, class_id]
    detections = outputs[0][0]
    
    boxes = []
    confidences = []
    class_ids = []
    
    # Get original image dimensions
    original_height, original_width = original_image.shape[:2]
    
    for detection in detections:
        x1, y1, x2, y2, conf, class_id = detection
        # Skip low confidence detections
        if conf < 0.5:
            continue
            
        # Scale boxes back to original image size
        x1 = int(x1 * original_width / input_width)
        y1 = int(y1 * original_height / input_height)
        x2 = int(x2 * original_width / input_width)
        y2 = int(y2 * original_height / input_height)
        
        boxes.append([x1, y1, x2, y2])
        confidences.append(float(conf))
        class_ids.append(int(class_id))
    
    return boxes, confidences, class_ids

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template("index.html")

    # Handle POST request
    image_data = None
    label = None
    message = None
    error = None

    if "image" not in request.files:
        return render_template("index.html", error="No image uploaded")

    file = request.files["image"]
    if file.filename == "":
        return render_template("index.html", error="No image selected")

    try:
        # Read image directly to memory
        img_bytes = file.read()
        img_np = np.frombuffer(img_bytes, np.uint8)
        original_img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

        if original_img is None:
            return render_template("index.html", error="Invalid image file")

        # Preprocess image for ONNX model
        img = preprocess_image(original_img.copy())
        
        # Run inference
        outputs = session.run(output_names, {input_names[0]: img})
        
        # Postprocess results
        boxes, confidences, class_ids = postprocess(outputs, original_img)
        
        # Convert to RGB for display
        display_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        
        if len(boxes) > 0:
            # Get the detection with highest confidence
            best_idx = np.argmax(confidences)
            best_box = boxes[best_idx]
            best_class_id = class_ids[best_idx]
            best_conf = confidences[best_idx]
            
            x1, y1, x2, y2 = best_box
            label = f"{class_names[best_class_id]}"
            
            # Draw bounding box and label
            cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 255, 0), 4)
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
            cv2.rectangle(display_img, (x1, y1 - text_height - 15), (x1 + text_width + 10, y1), (0, 255, 0), -1)
            cv2.putText(display_img, label, (x1 + 5, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        else:
            # No detection case
            label = "No Detection"
            message = "No cow detected. Please provide a clearer image of the cow's body."

            text = "No Cow Detected"
            font_scale = 2
            thickness = 4
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            x = (display_img.shape[1] - text_width) // 2
            y = (display_img.shape[0] + text_height) // 2
            cv2.putText(display_img, text, (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), thickness)

            suggestion = "Please provide a clearer image"
            font_scale_small = 0.8
            thickness_small = 2
            (sugg_width, sugg_height), _ = cv2.getTextSize(suggestion, cv2.FONT_HERSHEY_SIMPLEX, font_scale_small,
                                                           thickness_small)
            x_sugg = (display_img.shape[1] - sugg_width) // 2
            y_sugg = y + text_height + 20
            cv2.putText(display_img, suggestion, (x_sugg, y_sugg),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale_small, (0, 0, 255), thickness_small)

        # Convert image to base64 for displaying in HTML
        _, img_encoded = cv2.imencode('.jpg', cv2.cvtColor(display_img, cv2.COLOR_RGB2BGR))
        image_data = base64.b64encode(img_encoded).decode('utf-8')

    except Exception as e:
        error = f"Error processing image: {str(e)}"
        return render_template("index.html", error=error)

    return render_template("index.html", image_data=image_data, label=label, message=message, error=error)

# Similarly update the predict_realtime route...

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)

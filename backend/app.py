from flask import Flask, request, jsonify, Response, stream_with_context
from process_input import main
from flask_cors import CORS
import traceback
import torch
from torchvision import transforms, models
from PIL import Image
import cv2
import numpy as np
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)
CORS(app,resources={r"/api/*": {"origins": "http://localhost:5173"}}, supports_credentials=True)

# Load your PeopleCounterResNet50 model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "best_model.pth"

# Load object detection model (Faster R-CNN for this example)
detection_model = fasterrcnn_resnet50_fpn(pretrained=True).to(device)
detection_model.eval()

# Preprocessing function
def preprocess_image(image, channels=6):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    image_tensor = transform(image)
    # Simulating 6-channel input if required
    if channels == 6:
        image_tensor = torch.cat([image_tensor, image_tensor], dim=0)
    return image_tensor.unsqueeze(0).to(device)

# Process image
def process_image(image_path):
    image = Image.open(image_path).convert("RGB")
    img_tensor = preprocess_image(image)

    # Get bounding boxes from detection model
    img_cv2 = cv2.imread(image_path)
    img_cv2_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
    img_tensor = transforms.ToTensor()(img_cv2_rgb).unsqueeze(0).to(device)
    detections = detection_model(img_tensor)[0]

    # Filter "person" class (class_id = 1 for COCO dataset)
    boxes = detections['boxes'][detections['labels'] == 1]
    scores = detections['scores'][detections['labels'] == 1]

    # Count total number of people
    total_people = 0

    # Draw boxes on the image
    for box, score in zip(boxes, scores):
        if score > 0.5:  # Confidence threshold
            total_people += 1
            x1, y1, x2, y2 = map(int, box.cpu().detach().numpy())
            cv2.rectangle(img_cv2, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_cv2, f"{score:.2f}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Convert image to base64
    _, buffer = cv2.imencode('.jpg', img_cv2)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return img_base64, total_people

@app.route('/api/process', methods=['POST'])
def process():
    input_data = request.files.get('input') or request.form.get('input')
    input_type = request.form.get('inputType')
    overlapped = request.form.get('overlapped') == 'true'
    selected_model = request.form.get('selected_model')

    if not input_data or not input_type:
        return jsonify({'error': 'Missing input data or type'}), 400

    try:
        if input_type in ['video', 'url']:
            result = main(input_data, input_type)
            response = Response(
                stream_with_context(result),
                mimetype='multipart/x-mixed-replace; boundary=frame'
            )
            response.headers['Cache-Control'] = 'no-cache'
            response.headers['X-Accel-Buffering'] = 'no'
            return response
        elif input_type == 'url':
            # For video URL streams
            result = main(input_data, input_type)
            return Response(
                stream_with_context(result),
                mimetype='multipart/x-mixed-replace; boundary=frame'
            )
        elif input_type == 'image':
            # Image processing remains unchanged
            if overlapped:
                count = main(input_data.read(), input_type, overlapped=overlapped, selected_model=selected_model)
                count = count.tolist()
            else:
                count = main(input_data.read(), input_type, selected_model=selected_model)
            return jsonify(count), 200
        
        else:
            return jsonify({'error': 'Invalid input type'}), 400
    except Exception as e:
        print(f"Error processing request: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/api/process_image', methods=['POST'])
def process_image_route():
    input_data = request.files.get('input')
    if not input_data:
        return jsonify({'error': 'Missing input data'}), 400

    try:
        # Save the uploaded image to a temporary file
        temp_image_path = 'temp_image.jpg'
        input_data.save(temp_image_path)

        # Process the image and get the base64 string and total people count
        processed_image_base64, total_people = process_image(temp_image_path)

        return jsonify({'image': processed_image_base64, 'total_people': total_people}), 200
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)


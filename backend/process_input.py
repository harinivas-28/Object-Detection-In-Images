import cv2
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import math
import albumentations as A
from trained_models import models
from datetime import datetime
from flask import Response
from albumentations.pytorch import ToTensorV2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model():
    model = models.VGG16()
    model_path = "trained_models/vgg16_headcount.pth"
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval() 
    print("VGG16.Heavy Model loaded successfully")
    return model

def process_image(image, model):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(image).unsqueeze(0)
    input_tensor = input_tensor.to(device)
    with torch.no_grad():
        output = model(input_tensor)
    predicted_count = output.item() 
    print(f"Predicted Headcount: {predicted_count}")
    return math.ceil(predicted_count)

def process_video_stream(video, model):
    """Stream video frames to frontend."""
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        # Perform detection on the frame
        results = model(frame)
        detection_results = results.pandas().xyxy[0]
        for index, row in detection_results.iterrows():
            if row['name'] == 'person':
                x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'Person {row["confidence"]:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Convert frame to JPEG and yield as byte stream
        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            break
        frame_bytes = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

def process_url_stream(url, model):
    """Stream video from URL."""
    cap = cv2.VideoCapture(url)
    return process_video_stream(cap, model)

def load_video_model():
    try:
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    except Exception as e:
        print(f"Error loading YOLOv5 model: {str(e)}")
        return None
    print("YOLOv5 Model loaded successfully")
    return model

def load_overlapped_model():
    model = models.ResNet18()
    try:
        checkpoint = torch.load("trained_models/resnet18.pth", map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
    except Exception as e:
        print("Error:", e)  
    model.eval()
    model = model.to(device)
    print("ResNet18 Model loaded successfully")
    return model

def process_overlapped_image(image, model, target_size=(1223, 373)):
    image = np.array(image) 
    img = preprocess_image(image, target_size)
    img1, img2 = split_image(img)
    img1, img2 = transform_images(img1, img2)
    img_combined = torch.cat([img1, img2], dim=0).unsqueeze(0)
    img_combined = img_combined.to(device)
    with torch.no_grad():
        output = model(img_combined)
    outputs = output.cpu().numpy()
    return np.round(outputs,2)

def preprocess_image(image, target_size=(1223, 373)):
    """Resize image to target size"""
    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

def split_image(img):
    """Split the image into two equal halves"""
    width = img.shape[1]
    mid_point = width // 2
    img1 = img[:, :mid_point].copy()
    img2 = img[:, mid_point:].copy()
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    return img1, img2

def transform_images(img1, img2):
    """Apply normalization and convert to tensor"""
    transform = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    aug1 = transform(image=img1)
    aug2 = transform(image=img2)
    img1 = aug1['image']
    img2 = aug2['image']
    return img1, img2


def main(input_data, input_type, overlapped=False):
    print("Device: ",device)
    print("Input Type: ",input_type)
    if input_type == 'image':
        if isinstance(input_data, bytes):
            image = Image.open(BytesIO(input_data))
        else:
            image = Image.open(input_data)
        
        if overlapped:
            print(f"Overlapped Image: {overlapped}")
            model = load_overlapped_model()
            return process_overlapped_image(image, model)
        model = load_model()
        return process_image(image, model)
    elif input_type == 'video' or 'url':
        video_model = load_video_model()
        if input_type == 'video':
            if isinstance(input_data, bytes):
                temp_file = BytesIO(input_data)
                video = cv2.VideoCapture(temp_file.getvalue())
            else:
                video = cv2.VideoCapture(input_data)
            return process_video(video, video_model)
        elif input_type == 'url':
            return process_url(input_data, video_model)
    else:
        raise ValueError("Invalid input type")

if __name__ == "__main__":
    # Example usage:
    # result = main('test/img1.jpg', 'image', False)
    # print(f"Number of people detected: {result}")
    pass


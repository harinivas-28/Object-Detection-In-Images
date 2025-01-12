# Automated Classroom Attendance System Using Deep Learning and Transformers

## Overview
This project focuses on developing an **Automated Classroom Attendance System** leveraging **Deep Learning** and **Transformer models** to accurately count students in classrooms. By addressing challenges like **occlusions**, **overlapping individuals**, and **varied lighting conditions**, the system ensures high accuracy and robust performance in real-world scenarios.

## Abstract
The system automates the process of taking attendance by using advanced **Convolutional Neural Networks (CNNs)** and **Transformers** to detect and count students in classroom images or videos. A **custom dataset** was created for training and evaluation, ensuring adaptability to diverse scenarios. The solution integrates seamlessly with a **user-friendly interface** for real-time processing, enhancing classroom management efficiency.

## Objectives
1. Automate classroom attendance using deep learning techniques.
2. Develop a robust CNN-based model to predict student counts accurately.
3. Address challenges such as occlusions, overlapping, and varied densities.
4. Design a model that generalizes across diverse classroom setups.
5. Enable real-time deployment with edge devices for dynamic monitoring.
6. Create an intuitive interface for easy input and visualization of results.

## Problem Statement
Accurately counting individuals in crowded environments remains challenging due to:
- **Occlusions**
- **Overlapping individuals**
- **Varied densities**
- **Environmental factors** like lighting and shadows

Traditional approaches fail to generalize across scenarios, resulting in inaccurate predictions. This project aims to tackle these limitations by developing a scalable, efficient solution using deep learning techniques.

## Techniques and Tools Used
### Development Frameworks & Libraries
- **Deep Learning:**
  - PyTorch (Primary framework for ResNet18 implementation)
  - TensorFlow & Keras (Secondary framework for VGG16 implementation)
- **Data Processing:**
  - NumPy
  - Pandas
- **Visualization:**
  - Matplotlib
  - Seaborn

### Web Development Stack (MERN)
- **Frontend:** React.js
- **Backend:** Python Flask (RESTful API)
- **Version Control:** GitHub

### Model Architectures
- **ResNet18:**
  - Handles images with overlapping individuals using residual blocks.
- **VGG16:**
  - Optimized for non-overlapping images with dense layers and high feature extraction capabilities.
- **Planned:** Transformer-based models for advanced use cases.

## Dataset Details
- **Total Images:** 598
- **Source:** Custom dataset from classroom scenarios via CC cameras
- **Key Features:**
  - Multiple camera angles
  - Varied lighting conditions
  - Mixed occupancy levels (overlapped and non-overlapped scenarios)
- **Augmentation Techniques:**
  - Random rotation
  - Brightness variation
  - Horizontal flipping
  - Random cropping

## Implementation Highlights
1. **Data Preprocessing:**
   - Augmentation for dataset diversity
   - Normalization and resizing of images
2. **Model Training:**
   - Loss Function: Mean Squared Error (MSE)
   - Optimizer: Adam optimizer
   - Configuration: 50 epochs, batch size of 16
3. **Evaluation Metrics:**
   - MAE (Mean Absolute Error)
   - MSE (Mean Squared Error)
   - Density map visualization
4. **Deployment:**
   - Flask API for real-time inference
   - Edge device integration for on-the-spot monitoring
5. **User Interface:**
   - Options to upload images/videos or enter URLs
   - Real-time detection results with bounding boxes and counts

## Results
- **Accuracy:** 85-95% across diverse scenarios
- **Processing Time:** < 3 seconds per image
- **Scalability:** Supports up to 100 simultaneous users

## Future Scope
1. **Video Analysis Integration:**
   - Extend the system to analyze video sequences for real-time monitoring.
2. **Dataset Expansion:**
   - Include diverse classroom setups, lighting conditions, and crowd densities.
3. **Enhanced Attention Mechanisms:**
   - Implement advanced attention layers for improved accuracy.
4. **Cross-Domain Applications:**
   - Adapt the model for applications like wildlife counting, traffic monitoring, and disaster management.
5. **Performance Optimization:**
   - Reduce computational overhead for faster processing and scalability.


## How to Run the Project
1. Clone the repository: `git clone <repository-link>`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the Flask server: `python app.py`
4. Open the frontend: Navigate to the React.js project directory and run `npm start`


## License
[MIT License](LICENSE)

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/harinivas-28/count-app.git
    ```
2. Navigate to the project directory:
    ```sh
    cd count-app
    ```
3. Install dependencies:
    ```sh
    npm install
    ```
    ```sh
    cd backend
    pip install requirements.txt
    ```

## Usage

To start the frontend application, run:
```sh
npm run dev
```
To run the backend
```sh
cd backend
python app.py
```

## Contributing

Contributions are welcome! Please fork the repository and create a pull request with your changes.


## Contact

For any questions or feedback, please contact [harinivasg28704@gmail.com](mailto:harinivasg28704@gmail.com)
or [harinivas.ganjarla@gmail.com](mailto:harinivas.ganjarla@gmail.com)

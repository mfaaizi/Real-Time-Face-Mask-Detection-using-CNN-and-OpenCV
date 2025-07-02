😷 Real-Time Face Mask Detection using CNN and OpenCV

A real-time computer vision system that detects whether individuals are wearing face masks using a Convolutional Neural Network (CNN) and a webcam stream. This project helps monitor mask compliance in public areas and serves as an introductory application of deep learning for object classification.

🎯 Objective

To develop a lightweight CNN model capable of classifying faces as Mask or No Mask in real-time using webcam feed.

🚀 Features

Live face detection using OpenCV's Haar Cascade or DNN

CNN-based binary classification model (Mask vs No Mask)

Real-time inference and visual feedback with bounding boxes

Pre-trained model support for instant testing

Modular codebase with separate training and inference scripts

🧠 Tech Stack

Python

TensorFlow / Keras

OpenCV

NumPy

Matplotlib (for training visualization)

📁 Project Structure

bash
Copy
Edit
face_mask_detection/
│
├── train_mask_detector.py      # Script to train the CNN
├── detect_mask_video.py        # Real-time mask detection using webcam
├── mask_detector.model         # Trained CNN model
├── dataset/                    # Labeled images (with_mask / without_mask)
└── helpers/                    # Utilities for preprocessing and bounding boxes
🧪 How It Works

Detects faces in a webcam frame using OpenCV

Each face is passed to the trained CNN model

The model classifies the face as "Mask" or "No Mask"

Annotates the frame with labels and bounding boxes in real time

⚙️ How to Run

Clone the repository

Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
To train the model:

bash
Copy
Edit
python train_mask_detector.py
To run real-time detection:

bash
Copy
Edit
python detect_mask_video.py
📈 Results

Achieved high accuracy on a custom dataset of masked/unmasked faces

Efficient real-time detection on standard hardware (~20–30 FPS)

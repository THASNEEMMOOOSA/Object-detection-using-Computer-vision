📷 Object Detection using Computer Vision – ASL Sign Language Classifier
https://img.shields.io/badge/Python-3.8+-blue.svg
https://img.shields.io/badge/OpenCV-4.x-green.svg
https://img.shields.io/badge/PyTorch-1.x-red.svg
https://img.shields.io/badge/MediaPipe-0.10+-orange.svg

A real-time American Sign Language (ASL) classifier that detects hand gestures using MediaPipe landmarks and a PyTorch neural network.

📌 Overview
This project implements a complete pipeline for real-time ASL gesture recognition:

Data Collection – Capture hand images via webcam for any sign

Landmark Extraction – Use MediaPipe to extract 21 hand landmarks (x, y, z) → 63 features per image

Model Training – Train a neural network classifier on extracted landmarks

Real-Time Prediction – Predict gestures live from webcam feed

🛠️ Technologies Used
Technology	Purpose
Python 3.8+	Core programming language
OpenCV	Image capture, video processing, visualization
MediaPipe	Hand landmark detection (21 landmarks, 3D coordinates)
PyTorch	Neural network training and inference
NumPy	Data manipulation and array operations
scikit-learn	Train/test split
Tkinter	GUI for data collection
🚀 How It Works
1. Data Collection (data_collection.py)
Opens a Tkinter window asking for the ASL sign name

Opens webcam and starts capturing images

MediaPipe detects hand landmarks and draws them on the frame

Images are saved to dataset/[sign_name]/[0001.jpg, 0002.jpg, ...]

Captures 100 images per sign (configurable)

2. Landmark Extraction (extract_landmarks())
For each captured image, MediaPipe extracts 21 hand landmarks, each with:

x (normalized x-coordinate)

y (normalized y-coordinate)

z (depth relative to wrist)

Total features: 21 landmarks × 3 coordinates = 63 features per image

3. Model Training (train_model.py)
text
Input: 63 features (hand landmarks)
       ↓
Linear Layer (63 → 128) + ReLU
       ↓
Linear Layer (128 → 64) + ReLU
       ↓
Linear Layer (64 → num_classes)  # Softmax applied during inference
Training process:

Loads all images from dataset/ folders

Extracts landmarks for each image

Splits data into 80% training / 20% validation

Trains for 20 epochs using Adam optimizer and CrossEntropyLoss

Saves model as asl_model.pth

4. Real-Time Prediction (predict_live.py)
Opens webcam and processes each frame

Extracts hand landmarks using MediaPipe

Feeds 63 features into trained model

Displays predicted sign on screen in real time

Press q to quit

📂 Project Structure
text
Object-detection-using-Computer-vision/
│
├── data_collection.py          # Capture images for new signs
├── train_model.py              # Train classifier on captured data
├── predict_live.py             # Real-time prediction from webcam
│
├── dataset/                    # Training images (auto-created)
│   ├── A/
│   │   ├── 0001.jpg
│   │   ├── 0002.jpg
│   │   └── ...
│   ├── B/
│   └── ... (add any sign name)
│
├── asl_model.pth               # Saved trained model
│
├── requirements.txt            # Dependencies
└── README.md                   # This file
▶️ How to Run
Step 1: Clone the repository
bash
git clone https://github.com/THASNEEMMOOOSA/Object-detection-using-Computer-vision.git
cd Object-detection-using-Computer-vision
Step 2: Install dependencies
bash
pip install opencv-python mediapipe torch numpy scikit-learn pillow
Or use requirements.txt:

bash
pip install -r requirements.txt
Step 3: Collect training data
bash
python data_collection.py
Enter the ASL sign name (e.g., A, B, C, or hello, yes)

Webcam opens – show the sign with your hand

Automatically captures 100 images

Press q to quit early if needed

Repeat for each sign you want to train.

Step 4: Train the model
bash
python train_model.py
This will:

Load all images from dataset/

Extract hand landmarks

Train the neural network

Save asl_model.pth

Step 5: Run real-time prediction
bash
python predict_live.py
Webcam opens

Shows predicted sign in real time

Press q to quit

📊 Results
Input: 63 hand landmark features (21 points × x,y,z)

Architecture: 3-layer fully connected network

Training: 20 epochs, batch size 32, Adam optimizer

Accuracy: Achieves reliable classification for trained signs (add your actual accuracy here)

Real-time speed: ~30 FPS on standard webcam

🔧 Future Improvements
Add more training data for higher accuracy

Implement CNN on raw images instead of landmarks for richer features

Support two-handed signs

Add dynamic gestures (movement over time)

Deploy as a web app using Streamlit or Flask

Add GUI for easier data collection and prediction

📚 What I Learned
This project taught me:

Computer Vision: Image capture, preprocessing, real-time video processing with OpenCV

Hand Landmark Detection: Using MediaPipe to extract 3D hand coordinates

Feature Engineering: Converting spatial landmarks into a feature vector for classification

Deep Learning: Building, training, and deploying a PyTorch neural network

End-to-End Pipeline: From data collection to model deployment in a real-time application

🤝 Related Skills
This project demonstrates experience directly relevant to industrial image processing roles:

Skill	Application in This Project
Feature extraction	Hand landmarks from images
Classification	Neural network for gesture recognition
Real-time processing	Live webcam feed with <50ms latency
Data pipeline	Collection → training → inference
OpenCV	Image I/O, video capture, visualization
📬 Contact
Thasneem Moosa
📧 thasneemmoosa5000@gmail.com
🔗 LinkedIn
💻 GitHub

📄 License
This project is for educational purposes. Feel free to use and modify.

⭐ Show Your Support
If you found this project helpful, please give it a star on GitHub!

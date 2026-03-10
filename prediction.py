import cv2
import numpy as np
import torch
import torch.nn as nn
import mediapipe as mp
import os
# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model definition (same as training)
class ASLClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ASLClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(63, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.fc(x)

# Load label map (must match training)
label_map = {v: k for k, v in {
    name: idx for idx, name in enumerate(sorted(os.listdir("dataset")))
}.items()}

# Load model
model = ASLClassifier(num_classes=len(label_map)).to(device)
model.load_state_dict(torch.load("asl_model.pth", map_location=device))
model.eval()

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)

# Prediction function
def extract_landmarks(image):
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        return np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
    return None

# Webcam prediction loop
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    landmarks = extract_landmarks(frame)
    if landmarks is not None:
        input_tensor = torch.tensor(landmarks, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(input_tensor)
            _, predicted = torch.max(outputs, 1)
            label = label_map[predicted.item()]
            cv2.putText(frame, f"Prediction: {label}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("ASL Prediction", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

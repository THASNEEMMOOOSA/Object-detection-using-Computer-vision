import os
import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# Paths
DATASET_DIR = "dataset"  # dataset/A/*.jpg, dataset/B/*.jpg, etc.

# MediaPipe hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Extract hand landmarks from image
def extract_landmarks(image):
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1) as hands:
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if results.multi_hand_landmarks:
            return np.array([[lm.x, lm.y, lm.z] for lm in results.multi_hand_landmarks[0].landmark]).flatten()
    return None

# Custom dataset
class ASLDataset(Dataset):
    def __init__(self, root_dir):
        self.data = []
        self.labels = []
        self.label_map = {}
        self.load_data(root_dir)

    def load_data(self, root_dir):
        label_names = sorted(os.listdir(root_dir))
        for idx, label_name in enumerate(label_names):
            label_path = os.path.join(root_dir, label_name)
            if not os.path.isdir(label_path):
                continue
            self.label_map[label_name] = idx
            count = 0
            for img_file in os.listdir(label_path):
                img_path = os.path.join(label_path, img_file)
                image = cv2.imread(img_path)
                if image is None:
                    continue
                landmarks = extract_landmarks(image)
                if landmarks is not None:
                    self.data.append(landmarks)
                    self.labels.append(idx)
                    count += 1
            print(f"Loaded {count} samples for class '{label_name}'")


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32), self.labels[idx]

# Simple classifier
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

# Training
def train_model(dataset):
    train_data, val_data = train_test_split(dataset, test_size=0.2, random_state=42)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32)

    model = ASLClassifier(num_classes=len(dataset.label_map)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(20):
        model.train()
        total_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), torch.tensor(labels).to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), "asl_model.pth")
    print("Model saved as asl_model.pth")

# Run training
if __name__ == "__main__":
    dataset = ASLDataset(DATASET_DIR)
    if len(dataset) == 0:
        raise ValueError("No valid training data found.")
    train_model(dataset)

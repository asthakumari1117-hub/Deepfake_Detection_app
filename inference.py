import os
import cv2
import torch
import numpy as np
import torch.nn as nn
from facenet_pytorch import MTCNN
from torchvision import models, transforms

# -----------------------------
# DEVICE
# -----------------------------

device = torch.device("cpu")

# -----------------------------
# TRANSFORMER MODEL
# -----------------------------

class TransformerModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.embedding = nn.Linear(512, 256)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256,
            nhead=8,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )

        self.fc = nn.Linear(256, 2)

    def forward(self, x):

        x = self.embedding(x)

        x = x.unsqueeze(1)

        x = self.transformer(x)

        x = x.mean(dim=1)

        x = self.fc(x)

        return x

# -----------------------------
# LOAD TRAINED MODEL
# -----------------------------

model = TransformerModel()

model.load_state_dict(
    torch.load(
        "deepfake_transformer_model.pth",
        map_location=device
    )
)

model.eval()

print("Model Loaded Successfully")

# -----------------------------
# FACE DETECTOR
# -----------------------------

mtcnn = MTCNN(
    keep_all=True,
    device=device
)

# -----------------------------
# CNN FEATURE EXTRACTOR
# -----------------------------

cnn_model = models.resnet18(pretrained=True)

cnn_model = torch.nn.Sequential(
    *list(cnn_model.children())[:-1]
)

cnn_model.eval()

# -----------------------------
# IMAGE TRANSFORM
# -----------------------------

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# -----------------------------
# VIDEO PATH
# -----------------------------

video_path = "videos/original/000.mp4"

# Change this path later for testing

# -----------------------------
# OPEN VIDEO
# -----------------------------

cap = cv2.VideoCapture(video_path)

frame_count = 0

predictions = []

print("\nProcessing Video...\n")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Every 30th frame
    if frame_count % 30 == 0:

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        boxes, _ = mtcnn.detect(rgb)

        if boxes is not None:

            for box in boxes:

                x1, y1, x2, y2 = map(int, box)

                face = frame[y1:y2, x1:x2]

                if face.size == 0:
                    continue

                # -----------------------------
                # PREPROCESS FACE
                # -----------------------------

                face = cv2.resize(
                    face,
                    (224, 224)
                )

                face = cv2.cvtColor(
                    face,
                    cv2.COLOR_BGR2RGB
                )

                tensor = transform(face)

                tensor = tensor.unsqueeze(0)

                tensor = tensor.to(device)

                # -----------------------------
                # CNN FEATURE EXTRACTION
                # -----------------------------

                with torch.no_grad():

                    features = cnn_model(tensor)

                features = features.squeeze()

                features = features.unsqueeze(0)

                features = features.to(device)

                # -----------------------------
                # TRANSFORMER PREDICTION
                # -----------------------------

                with torch.no_grad():

                    output = model(features)

                    print("\n-------------------")
                    print("Raw Output:", output)

                    prediction = torch.argmax(
                        output,
                        dim=1
                    )

                    print(
                        "Prediction Tensor:",
                        prediction
                    )

                    print(
                        "Prediction Value:",
                        prediction.item()
                    )

                    # 0 = REAL
                    # 1 = FAKE

                    if prediction.item() == 0:

                        print("Frame Result: REAL")

                    else:

                        print("Frame Result: FAKE")

                    predictions.append(
                        prediction.item()
                    )

        print(f"\nProcessed Frame: {frame_count}")

    frame_count += 1

cap.release()

# -----------------------------
# FINAL RESULT
# -----------------------------

fake_count = predictions.count(1)

real_count = predictions.count(0)

total_predictions = len(predictions)

print("\n===================")
print("FINAL RESULTS")
print("===================")

print("Total Predictions:", total_predictions)
print("Real Predictions:", real_count)
print("Fake Predictions:", fake_count)

# Avoid division by zero

if total_predictions > 0:

    fake_percentage = (
        fake_count / total_predictions
    ) * 100

    real_percentage = (
        real_count / total_predictions
    ) * 100

    print(
        f"\nFake Percentage: {fake_percentage:.2f}%"
    )

    print(
        f"Real Percentage: {real_percentage:.2f}%"
    )

    # Final Decision

    if fake_percentage > 60:

        print("\nFINAL RESULT: FAKE VIDEO")

    else:

        print("\nFINAL RESULT: REAL VIDEO")

else:

    print("\nNo faces detected in video.")
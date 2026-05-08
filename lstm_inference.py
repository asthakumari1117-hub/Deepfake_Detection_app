import cv2
import torch
import numpy as np

from PIL import Image
from torchvision import transforms, models
from facenet_pytorch import MTCNN

import torch.nn as nn

# -----------------------------
# DEVICE
# -----------------------------

device = torch.device("cpu")

# -----------------------------
# LSTM MODEL
# -----------------------------

class DeepfakeLSTM(nn.Module):

    def __init__(self):

        super().__init__()

        self.lstm = nn.LSTM(

            input_size=512,

            hidden_size=128,

            num_layers=2,

            batch_first=True

        )

        self.fc = nn.Linear(
            128,
            2
        )

    def forward(self, x):

        output, (hidden, cell) = self.lstm(x)

        x = hidden[-1]

        x = self.fc(x)

        return x

# -----------------------------
# LOAD TRAINED MODEL
# -----------------------------

model = DeepfakeLSTM()

model.load_state_dict(

    torch.load(
        "lstm_deepfake_model.pth",
        map_location=device
    )

)

model.eval()

print("LSTM Model Loaded")

# -----------------------------
# RESNET18 FEATURE EXTRACTOR
# -----------------------------

cnn_model = models.resnet18(pretrained=True)

cnn_model = torch.nn.Sequential(
    *list(cnn_model.children())[:-1]
)

cnn_model.eval()

# -----------------------------
# FACE DETECTOR
# -----------------------------

mtcnn = MTCNN(
    keep_all=True,
    device=device
)

# -----------------------------
# TRANSFORM
# -----------------------------

transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.ToTensor(),

])

# -----------------------------
# VIDEO PATH
# -----------------------------

video_path = "videos/Original/000.mp4"

# Change for testing

# -----------------------------
# OPEN VIDEO
# -----------------------------

cap = cv2.VideoCapture(video_path)

frame_count = 0

sequence = []

predictions = []

print("\nProcessing Video...")

# -----------------------------
# PROCESS VIDEO
# -----------------------------

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

                face = cv2.cvtColor(
                    face,
                    cv2.COLOR_BGR2RGB
                )

                face = Image.fromarray(face)

                face = transform(face)

                face = face.unsqueeze(0)

                # CNN Features
                with torch.no_grad():

                    feature = cnn_model(face)

                feature = feature.squeeze().numpy()

                sequence.append(feature)

                print(f"Processed Frame: {frame_count}")

                # Predict after 10 frames
                if len(sequence) == 10:

                    sequence_tensor = torch.tensor(

                        [sequence],

                        dtype=torch.float32

                    )

                    with torch.no_grad():

                        output = model(sequence_tensor)

                        prediction = torch.argmax(
                            output,
                            dim=1
                        ).item()

                    predictions.append(prediction)

                    print("\nPrediction:", prediction)

                    if prediction == 0:

                        print("Sequence Result: REAL")

                    else:

                        print("Sequence Result: FAKE")

                    sequence = []

    frame_count += 1

cap.release()

# -----------------------------
# FINAL RESULT
# -----------------------------

real_count = predictions.count(0)

fake_count = predictions.count(1)

print("\n===================")
print("FINAL RESULTS")
print("===================")

print("Real Sequences:", real_count)

print("Fake Sequences:", fake_count)

total = len(predictions)

if total > 0:

    real_percentage = (real_count / total) * 100

    fake_percentage = (fake_count / total) * 100

    print(f"\nReal Percentage: {real_percentage:.2f}%")

    print(f"Fake Percentage: {fake_percentage:.2f}%")

    if fake_count > real_count:

        print("\nFINAL RESULT: FAKE VIDEO")

    else:

        print("\nFINAL RESULT: REAL VIDEO")

else:

    print("No valid sequences found")
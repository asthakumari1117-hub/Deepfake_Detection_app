import streamlit as st
import tempfile
import cv2
import torch
import numpy as np
import torch.nn as nn

from facenet_pytorch import MTCNN
from torchvision import models, transforms

# -----------------------------
# PAGE CONFIG
# -----------------------------

st.set_page_config(
    page_title="Deepfake Detector",
    layout="centered"
)

st.title("🎭 Deepfake Video Detection")

# -----------------------------
# DEVICE
# -----------------------------

device = torch.device("cpu")

# -----------------------------
# LSTM MODEL
# -----------------------------

class LSTMModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=128,
            num_layers=2,
            batch_first=True
        )

        self.fc = nn.Linear(128, 2)

    def forward(self, x):

        lstm_out, (hidden, cell) = self.lstm(x)

        output = self.fc(hidden[-1])

        return output

# -----------------------------
# LOAD MODEL
# -----------------------------

@st.cache_resource
def load_model():

    model = LSTMModel()

    model.load_state_dict(
        torch.load(
            "lstm_deepfake_model.pth",
            map_location=device
        )
    )

    model.eval()

    return model

model = load_model()

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
# VIDEO UPLOAD
# -----------------------------

uploaded_file = st.file_uploader(
    "Upload Video",
    type=["mp4", "avi", "mov"]
)

# -----------------------------
# PREDICTION
# -----------------------------

if uploaded_file is not None:

    st.video(uploaded_file)

    if st.button("Detect Deepfake"):

        with st.spinner("Processing Video..."):

            # Save Uploaded Video
            temp_video = tempfile.NamedTemporaryFile(
                delete=False
            )

            temp_video.write(
                uploaded_file.read()
            )

            video_path = temp_video.name

            cap = cv2.VideoCapture(video_path)

            frame_count = 0

            sequence = []

            real_sequences = 0

            fake_sequences = 0

            while True:

                ret, frame = cap.read()

                if not ret:
                    break

                # Every 30th Frame
                if frame_count % 10 == 0:

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

                            with torch.no_grad():

                                features = cnn_model(tensor)

                            features = features.squeeze()

                            sequence.append(
                                features.numpy()
                            )

                            # Sequence Length = 10
                            if len(sequence) == 10:

                                sequence_array = np.array(
                                    sequence
                                )

                                sequence_tensor = torch.tensor(
                                    sequence_array,
                                    dtype=torch.float32
                                )

                                sequence_tensor = sequence_tensor.unsqueeze(0)

                                with torch.no_grad():

                                    output = model(sequence_tensor)

                                    prediction = torch.argmax(
                                        output,
                                        dim=1
                                    ).item()

                                if prediction == 0:

                                    real_sequences += 1

                                else:

                                    fake_sequences += 1

                                # Reset Sequence
                                sequence = []

                frame_count += 1

            cap.release()

            # -----------------------------
            # FINAL RESULTS
            # -----------------------------

            total = real_sequences + fake_sequences

            real_percent = 0
            fake_percent = 0

            if total > 0:

                real_percent = (
                    real_sequences / total
                ) * 100

                fake_percent = (
                    fake_sequences / total
                ) * 100

            st.subheader("Results")

            st.write(
                f"Real Sequences: {real_sequences}"
            )

            st.write(
                f"Fake Sequences: {fake_sequences}"
            )

            st.write(
                f"Real Percentage: {real_percent:.2f}%"
            )

            st.write(
                f"Fake Percentage: {fake_percent:.2f}%"
            )

            # -----------------------------
            # IMPROVED FINAL RESULT
            # -----------------------------

            if total > 0:

                fake_ratio = fake_sequences / total

                if fake_ratio > 0.75:

                    st.error(
                        "🚨 FAKE VIDEO DETECTED"
                    )

                elif fake_ratio < 0.25:

                    st.success(
                        "✅ REAL VIDEO DETECTED"
                    )

                else:

                    st.warning(
                        "⚠️ SUSPICIOUS / UNCERTAIN VIDEO"
                    )

            else:

                st.warning(
                    "No face sequences detected"
                )
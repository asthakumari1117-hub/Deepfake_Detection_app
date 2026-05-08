import os
import torch
import numpy as np

from torchvision import transforms, models
from PIL import Image

# -----------------------------
# SETTINGS
# -----------------------------

SEQUENCE_LENGTH = 10

IMAGE_SIZE = 224

# -----------------------------
# TRANSFORM
# -----------------------------

transform = transforms.Compose([

    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.ToTensor(),

])

# -----------------------------
# LOAD RESNET18
# -----------------------------

cnn_model = models.resnet18(pretrained=True)

cnn_model = torch.nn.Sequential(
    *list(cnn_model.children())[:-1]
)

cnn_model.eval()

# -----------------------------
# FEATURE EXTRACTION
# -----------------------------

def extract_feature(image_path):

    image = Image.open(image_path).convert("RGB")

    image = transform(image)

    image = image.unsqueeze(0)

    with torch.no_grad():

        feature = cnn_model(image)

    feature = feature.squeeze()

    return feature.numpy()

# -----------------------------
# CREATE SEQUENCES
# -----------------------------

def create_sequences(folder, label):

    sequences = []

    labels = []

    files = sorted(os.listdir(folder))

    current_sequence = []

    for file in files:

        path = os.path.join(folder, file)

        feature = extract_feature(path)

        current_sequence.append(feature)

        if len(current_sequence) == SEQUENCE_LENGTH:

            sequences.append(current_sequence)

            labels.append(label)

            current_sequence = []

    return sequences, labels

# -----------------------------
# REAL DATA
# -----------------------------

real_folder = "processed_faces/real"

real_sequences, real_labels = create_sequences(
    real_folder,
    0
)

# -----------------------------
# FAKE DATA
# -----------------------------

fake_folder = "processed_faces/fake"

fake_sequences, fake_labels = create_sequences(
    fake_folder,
    1
)

# -----------------------------
# COMBINE
# -----------------------------

all_sequences = real_sequences + fake_sequences

all_labels = real_labels + fake_labels

# -----------------------------
# SAVE
# -----------------------------

np.save(

    "lstm_features.npy",

    np.array(all_sequences)

)

np.save(

    "lstm_labels.npy",

    np.array(all_labels)

)

print("\nSequences Created Successfully")

print("Total Sequences:", len(all_sequences))

print("Sequence Shape:", np.array(all_sequences).shape)
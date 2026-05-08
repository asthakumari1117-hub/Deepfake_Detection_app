import numpy as np
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

# -----------------------------
# DEVICE
# -----------------------------

device = torch.device("cpu")

# -----------------------------
# LOAD DATA
# -----------------------------

X = np.load("lstm_features.npy")

y = np.load("lstm_labels.npy")

print("Features Shape:", X.shape)

print("Labels Shape:", y.shape)

# -----------------------------
# TRAIN TEST SPLIT
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.2,

    random_state=42,

    shuffle=True

)

print("\nTraining Samples:", len(X_train))

print("Testing Samples:", len(X_test))

# -----------------------------
# CONVERT TO TENSORS
# -----------------------------

X_train = torch.tensor(
    X_train,
    dtype=torch.float32
)

X_test = torch.tensor(
    X_test,
    dtype=torch.float32
)

y_train = torch.tensor(
    y_train,
    dtype=torch.long
)

y_test = torch.tensor(
    y_test,
    dtype=torch.long
)

# -----------------------------
# DATALOADER
# -----------------------------

train_dataset = TensorDataset(
    X_train,
    y_train
)

test_dataset = TensorDataset(
    X_test,
    y_test
)

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False
)

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
# MODEL
# -----------------------------

model = DeepfakeLSTM().to(device)

print("\nModel Ready")

# -----------------------------
# LOSS + OPTIMIZER
# -----------------------------

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(

    model.parameters(),

    lr=0.0001

)

# -----------------------------
# TRAINING
# -----------------------------

epochs = 10

print("\nTraining Started")

for epoch in range(epochs):

    model.train()

    total_loss = 0

    correct = 0

    total = 0

    for sequences, labels in train_loader:

        sequences = sequences.to(device)

        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(sequences)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total

    print(f"\nEpoch {epoch+1}")

    print("Loss:", total_loss)

    print("Training Accuracy:", accuracy)

# -----------------------------
# TESTING
# -----------------------------

model.eval()

correct = 0

total = 0

with torch.no_grad():

    for sequences, labels in test_loader:

        sequences = sequences.to(device)

        labels = labels.to(device)

        outputs = model(sequences)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

test_accuracy = 100 * correct / total

print("\nFINAL TEST ACCURACY:", test_accuracy)

# -----------------------------
# SAVE MODEL
# -----------------------------

torch.save(

    model.state_dict(),

    "lstm_deepfake_model.pth"

)

print("\nMODEL SAVED SUCCESSFULLY")
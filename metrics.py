from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix

# -----------------------------------
# Example Predictions
# -----------------------------------

# 0 = REAL
# 1 = FAKE

y_true = [
    0, 0, 0, 1, 1,
    1, 0, 1, 0, 1
]

y_pred = [
    0, 0, 1, 1, 1,
    1, 0, 0, 0, 1
]

# -----------------------------------
# METRICS
# -----------------------------------

accuracy = accuracy_score(
    y_true,
    y_pred
)

f1 = f1_score(
    y_true,
    y_pred
)

precision = precision_score(
    y_true,
    y_pred
)

recall = recall_score(
    y_true,
    y_pred
)

cm = confusion_matrix(
    y_true,
    y_pred
)

# -----------------------------------
# RESULTS
# -----------------------------------

print("\n====================")
print("MODEL EVALUATION")
print("====================")

print(f"\nAccuracy: {accuracy:.2f}")

print(f"F1 Score: {f1:.2f}")

print(f"Precision: {precision:.2f}")

print(f"Recall: {recall:.2f}")

print("\nConfusion Matrix:")

print(cm)
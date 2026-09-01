import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping


# ==========================================
# Dataset location
# ==========================================

project_root = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

base_path = os.path.join(
    project_root,
    "dataset",
    "sequences"
)

labels = ["smash", "drop", "clear"]

label_map = {
    label: num
    for num, label in enumerate(labels)
}


# ==========================================
# Load dataset
# ==========================================

X = []
y = []

for label in labels:

    folder = os.path.join(
        base_path,
        label
    )

    for file in os.listdir(folder):

        if file.endswith(".npy"):

            sequence = np.load(
                os.path.join(folder, file)
            )

            # Only accept correct sequence shape
            if sequence.shape == (30, 99):

                X.append(sequence)
                y.append(label_map[label])


X = np.array(X, dtype="float32")
y = np.array(y)


print("Total dataset:", X.shape)
print("Labels:", y.shape)


# ==========================================
# Convert labels
# ==========================================

y_categorical = to_categorical(
    y,
    num_classes=3
)


# ==========================================
# Train / Validation / Test split
# ==========================================

# First split:
# 85% Train + Validation
# 15% Test

X_train_val, X_test, y_train_val, y_test, y_train_val_raw, y_test_raw = train_test_split(
    X,
    y_categorical,
    y,
    test_size=0.15,
    random_state=42,
    stratify=y
)


# Second split:
# 82.35% of 85% ≈ 70% total
# 17.65% of 85% ≈ 15% total

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val,
    y_train_val,
    test_size=0.1765,
    random_state=42,
    stratify=np.argmax(y_train_val, axis=1)
)


print("\n========================================")
print("DATASET SPLIT")
print("========================================")

print("Training:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)


# ==========================================
# Build LSTM model
# ==========================================

model = Sequential()

model.add(
    Bidirectional(
        LSTM(
            64,
            return_sequences=False
        ),
        input_shape=(
            X.shape[1],
            X.shape[2]
        )
    )
)

model.add(
    Dense(
        32,
        activation="relu"
    )
)

model.add(
    Dense(
        3,
        activation="softmax"
    )
)


# ==========================================
# Compile
# ==========================================

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


# ==========================================
# Early stopping
# ==========================================

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)


# ==========================================
# Train
# ==========================================

print("\n========================================")
print("TRAINING")
print("========================================")

history = model.fit(
    X_train,
    y_train,
    epochs=100,
    validation_data=(
        X_val,
        y_val
    ),
    callbacks=[early_stop]
)


# ==========================================
# Save model
# ==========================================

models_dir = os.path.join(
    project_root,
    "models"
)

os.makedirs(
    models_dir,
    exist_ok=True
)

model_path = os.path.join(
    models_dir,
    "stroke_classifier.keras"
)

model.save(model_path)

print("\nModel saved to:")
print(model_path)


# ==========================================
# FINAL TEST
# ==========================================

print("\n========================================")
print("FINAL TEST SET EVALUATION")
print("========================================")

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print(
    f"Test Loss: {test_loss:.4f}"
)

print(
    f"Test Accuracy: {test_accuracy:.4f}"
    f" ({test_accuracy * 100:.2f}%)"
)


# ==========================================
# Predictions
# ==========================================

y_probability = model.predict(
    X_test,
    verbose=0
)

y_pred = np.argmax(
    y_probability,
    axis=1
)

y_true = y_test_raw


# ==========================================
# Overall metrics
# ==========================================

accuracy = accuracy_score(
    y_true,
    y_pred
)

precision = precision_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)


print("\n========================================")
print("OVERALL PERFORMANCE")
print("========================================")

print(
    f"Accuracy : {accuracy:.4f}"
)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall   : {recall:.4f}"
)

print(
    f"F1-score : {f1:.4f}"
)


# ==========================================
# Per-class results
# ==========================================

print("\n========================================")
print("PER-CLASS PERFORMANCE")
print("========================================")

report = classification_report(
    y_true,
    y_pred,
    target_names=labels,
    digits=4,
    zero_division=0
)

print(report)


# ==========================================
# Confusion Matrix
# ==========================================

cm = confusion_matrix(
    y_true,
    y_pred
)

print("\n========================================")
print("CONFUSION MATRIX")
print("========================================")

print(cm)


# ==========================================
# Plot confusion matrix
# ==========================================

plt.figure(
    figsize=(7, 6)
)

plt.imshow(cm)

plt.title(
    "Badminton Stroke Classification\nConfusion Matrix"
)

plt.xlabel(
    "Predicted Label"
)

plt.ylabel(
    "True Label"
)

plt.xticks(
    range(len(labels)),
    labels
)

plt.yticks(
    range(len(labels)),
    labels
)


# Add numbers

for i in range(len(labels)):

    for j in range(len(labels)):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )


plt.colorbar()

plt.tight_layout()


# ==========================================
# Save confusion matrix
# ==========================================

results_dir = os.path.join(
    project_root,
    "evaluation",
    "results"
)

os.makedirs(
    results_dir,
    exist_ok=True
)

cm_path = os.path.join(
    results_dir,
    "confusion_matrix.png"
)

plt.savefig(
    cm_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


print("\nConfusion matrix saved to:")
print(cm_path)

print("\n========================================")
print("EVALUATION COMPLETE")
print("========================================")
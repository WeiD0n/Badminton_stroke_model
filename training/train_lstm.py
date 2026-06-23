import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping

# Dataset location
base_path = "dataset/sequences"

labels = ["smash", "drop", "clear"]
label_map = {label: num for num, label in enumerate(labels)}

X = []
y = []

# Load data
for label in labels:

    folder = os.path.join(base_path, label)

    for file in os.listdir(folder):

        if file.endswith(".npy"):

            sequence = np.load(os.path.join(folder, file))

            X.append(sequence)
            y.append(label_map[label])

# Pad sequences to same length
X = pad_sequences(
    X,
    padding="post",
    dtype="float32"
)

# Convert labels
y = to_categorical(y)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# Build LSTM model
model = Sequential()

model.add(
    Bidirectional(
        LSTM(
            64,
            return_sequences=False
        ),
        input_shape=(X.shape[1], X.shape[2])
    )
)

model.add(Dense(32, activation="relu"))
model.add(Dense(3, activation="softmax"))

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Early stopping
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

# Train
history = model.fit(
    X_train,
    y_train,
    epochs=100,
    validation_data=(X_test, y_test),
    callbacks=[early_stop]
)

# Save model
model.save("models/stroke_classifier.keras")

print("Model saved!")
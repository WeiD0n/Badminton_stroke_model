"""
Train the production badminton stroke classifier using the E06 configuration.

Input: dataset/sequences_tracked/{clear,drop,smash}/*.npy
Each sequence must contain 30 frames x 99 MediaPipe pose features.

Important for the web application:
- Class order is ["clear", "drop", "smash"].
- The saved model is models/stroke_classifier.keras.
- Inference must normalize inputs using models/normalisation_stats.npz
  before calling model.predict().
"""

import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split

SEED = 42
CLASSES = ["clear", "drop", "smash"] 
NUM_FRAMES = 30
NUM_FEATURES = 99

TEST_SIZE = 0.15
VALIDATION_FRACTION_OF_REMAINDER = 0.1765  
BATCH_SIZE = 3
EPOCHS = 100
LEARNING_RATE = 0.001
PATIENCE = 12

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEQUENCES_DIR = PROJECT_ROOT / "dataset" / "sequences_tracked"
MODELS_DIR = PROJECT_ROOT / "models"
EVALUATION_DIR = PROJECT_ROOT / "evaluation"

# Keep the standard filename expected by the existing application.
MODEL_PATH = MODELS_DIR / "stroke_classifier.keras"
BEST_MODEL_PATH = MODELS_DIR / "stroke_classifier_best.keras"
NORMALISATION_PATH = MODELS_DIR / "normalisation_stats.npz"


def set_seeds():
    os.environ.setdefault("PYTHONHASHSEED", str(SEED))
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)


def load_dataset():
    """Load valid (30, 99) sequences; label is determined by parent folder."""
    X, y, source_files = [], [], []
    skipped_files = []
    class_counts = {name: 0 for name in CLASSES}

    for label, class_name in enumerate(CLASSES):
        class_dir = SEQUENCES_DIR / class_name
        if not class_dir.exists():
            print(f"WARNING: Missing class folder: {class_dir}")
            continue

        for file_path in sorted(class_dir.glob("*.npy")):
            try:
                sequence = np.asarray(
                    np.load(file_path, allow_pickle=False), dtype=np.float32
                )
                if sequence.shape != (NUM_FRAMES, NUM_FEATURES):
                    raise ValueError(
                        f"Expected {(NUM_FRAMES, NUM_FEATURES)}, got {sequence.shape}. "
                        "Resample/extract this sequence before training."
                    )
                if not np.isfinite(sequence).all():
                    raise ValueError("Sequence contains NaN or infinite values.")

                X.append(sequence)
                y.append(label)
                source_files.append(str(file_path.relative_to(PROJECT_ROOT)))
                class_counts[class_name] += 1
            except Exception as exc:
                skipped_files.append((file_path, str(exc)))
                print(f"SKIPPED: {file_path.name} — {exc}")

    if not X:
        raise RuntimeError(f"No valid sequences found in {SEQUENCES_DIR}")

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int32)

    print("\\n================ DATASET SUMMARY ================")
    for name in CLASSES:
        print(f"{name:>8}: {class_counts[name]} sequences")
    print(f"Total usable sequences: {len(X)}")
    print(f"X shape: {X.shape}")
    print(f"Skipped files: {len(skipped_files)}")

    missing = [name for name, count in class_counts.items() if count < 3]
    if missing:
        raise RuntimeError(
            f"Need at least 3 valid sequences per class for stratified splitting: {missing}"
        )
    return X, y, source_files


def build_model():
    """E06 architecture and compile settings."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(NUM_FRAMES, NUM_FEATURES)),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dropout(0.30),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(len(CLASSES), activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def save_training_plot(history, key, val_key, title, ylabel, output_path):
    epochs = range(1, len(history.history[key]) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history.history[key], label=f"Training {ylabel.lower()}")
    plt.plot(epochs, history.history[val_key], label=f"Validation {ylabel.lower()}")
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main():
    set_seeds()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)

    X, y, source_files = load_dataset()
    X_pool, X_test, y_pool, y_test, files_pool, files_test = train_test_split(
        X, y, source_files, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )
    X_train, X_val, y_train, y_val, files_train, files_val = train_test_split(
        X_pool, y_pool, files_pool,
        test_size=VALIDATION_FRACTION_OF_REMAINDER,
        random_state=SEED,
        stratify=y_pool,
    )

    print("\\n================ DATA SPLIT ================")
    print(f"Training: {len(X_train)}")
    print(f"Validation: {len(X_val)}")
    print(f"Testing: {len(X_test)}")
    print(f"Class order: {CLASSES}")

    mean = np.mean(X_train, axis=(0, 1), keepdims=True).astype(np.float32)
    std = np.std(X_train, axis=(0, 1), keepdims=True).astype(np.float32)
    std[std < 1e-7] = 1.0

    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std

    np.savez(
        NORMALISATION_PATH,
        mean=mean.reshape(NUM_FEATURES),
        std=std.reshape(NUM_FEATURES),
        classes=np.asarray(CLASSES),
    )
    print(f"Saved inference normalization stats: {NORMALISATION_PATH}")

    manifest_path = EVALUATION_DIR / "data_split_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        f.write("file,split,class\\n")
        for split_name, file_list, labels in [
            ("train", files_train, y_train),
            ("validation", files_val, y_val),
            ("test", files_test, y_test),
        ]:
            for filename, label in zip(file_list, labels):
                f.write(f'"{filename}",{split_name},{CLASSES[int(label)]}\\n')

    model = build_model()
    model.summary()
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=PATIENCE,
            restore_best_weights=True, verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(BEST_MODEL_PATH), monitor="val_loss",
            save_best_only=True, verbose=1,
        ),
    ]

    print("TRAINING STARTED")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
        shuffle=True,
    )

    # Save final model to the standard path used by the application.
    model.save(MODEL_PATH)
    print(f"Saved application model: {MODEL_PATH}")
    print(f"Saved best checkpoint: {BEST_MODEL_PATH}")

    save_training_plot(
        history, "accuracy", "val_accuracy",
        "E06 Training and Validation Accuracy", "Accuracy",
        EVALUATION_DIR / "training_accuracy.png",
    )
    save_training_plot(
        history, "loss", "val_loss",
        "E06 Training and Validation Loss", "Loss",
        EVALUATION_DIR / "training_loss.png",
    )

    # One final evaluation on held-out test data.
    test_loss, test_accuracy = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE, verbose=0)
    probabilities = model.predict(X_test, batch_size=BATCH_SIZE, verbose=0)
    y_pred = np.argmax(probabilities, axis=1)

    report = classification_report(
        y_test, y_pred, labels=list(range(len(CLASSES))),
        target_names=CLASSES, digits=4, zero_division=0,
    )
    cm = confusion_matrix(y_test, y_pred, labels=list(range(len(CLASSES))))
    print(f"\\nTest loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")
    print("\\nClassification report:\\n" + report)
    print("Confusion matrix (rows=actual, columns=predicted):")
    print(cm)

    report_path = EVALUATION_DIR / "classification_report.txt"
    report_path.write_text(
        f"Test loss: {test_loss:.6f}\\nTest accuracy: {test_accuracy:.6f}\\n"
        f"Class order: {', '.join(CLASSES)}\\n\\n{report}\\n"
        f"Confusion matrix (rows=actual, columns=predicted):\\n{np.array2string(cm)}",
        encoding="utf-8",
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=[name.title() for name in CLASSES]
    )
    display.plot(values_format="d")
    plt.title("E06 Test Set Confusion Matrix")
    plt.tight_layout()
    plt.savefig(EVALUATION_DIR / "confusion_matrix.png", dpi=150)
    plt.close()

    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, labels=list(range(len(CLASSES))), zero_division=0
    )
    metrics_path = EVALUATION_DIR / "per_class_metrics.csv"
    with metrics_path.open("w", encoding="utf-8", newline="") as f:
        f.write("class,precision,recall,f1_score,support\\n")
        for i, name in enumerate(CLASSES):
            f.write(f"{name},{precision[i]:.6f},{recall[i]:.6f},{f1[i]:.6f},{int(support[i])}\\n")

    print("\\n================ OUTPUTS ================")
    for output in [
        MODEL_PATH, BEST_MODEL_PATH, NORMALISATION_PATH, manifest_path,
        EVALUATION_DIR / "training_accuracy.png",
        EVALUATION_DIR / "training_loss.png",
        report_path, EVALUATION_DIR / "confusion_matrix.png", metrics_path,
    ]:
        print(output)
    print("\\nTraining finished. The app must load stroke_classifier.keras and apply normalisation_stats.npz.")


if __name__ == "__main__":
    main()

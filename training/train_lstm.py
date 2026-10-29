import os
import random
import json
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
NUM_POSE_FEATURES = 99
NUM_BIOMECHANICAL_FEATURES = 12
NUM_FEATURES = NUM_POSE_FEATURES + NUM_BIOMECHANICAL_FEATURES

TEST_SIZE = 0.15
VALIDATION_FRACTION_OF_REMAINDER = 0.1765
BATCH_SIZE = 3
EPOCHS = 100
LEARNING_RATE = 0.001
PATIENCE = 12

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEQUENCES_DIR = PROJECT_ROOT / "dataset" / "sequences_tracked"
MODELS_DIR = PROJECT_ROOT / "models"
EVALUATION_DIR = PROJECT_ROOT / "evaluation" / "biomechanical"

MODEL_PATH = MODELS_DIR / "stroke_classifier_biomechanical.keras"
BEST_MODEL_PATH = MODELS_DIR / "stroke_classifier_biomechanical_best.keras"
NORMALISATION_PATH = MODELS_DIR / "normalisation_stats_biomechanical.npz"
CONFIG_PATH = EVALUATION_DIR / "training_config.json"

# MediaPipe Pose landmark indices.
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_ELBOW, RIGHT_ELBOW = 13, 14
LEFT_WRIST, RIGHT_WRIST = 15, 16
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_KNEE, RIGHT_KNEE = 25, 26
LEFT_ANKLE, RIGHT_ANKLE = 27, 28
EPSILON = 1e-7


def set_seeds():
    os.environ.setdefault("PYTHONHASHSEED", str(SEED))
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)


def calculate_angle(a, b, c):
    """Calculate angle ABC in degrees from 3D landmark coordinates."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    c = np.asarray(c, dtype=np.float32)
    ba = a - b
    bc = c - b
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba < EPSILON or norm_bc < EPSILON:
        return np.nan
    cosine = np.dot(ba, bc) / (norm_ba * norm_bc)
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def calculate_biomechanical_features(sequence):
    """
    Produce 12 features per frame:
    8 joint angles, 2 coordinate-system-dependent torso lean proxies,
    and 2 frame-to-frame elbow-angle changes.
    """
    sequence = np.asarray(sequence, dtype=np.float32)
    if sequence.shape != (NUM_FRAMES, NUM_POSE_FEATURES):
        raise ValueError(
            f"Expected pose sequence {(NUM_FRAMES, NUM_POSE_FEATURES)}, "
            f"got {sequence.shape}"
        )

    landmarks = sequence.reshape(NUM_FRAMES, 33, 3)
    bio = np.zeros((NUM_FRAMES, NUM_BIOMECHANICAL_FEATURES), dtype=np.float32)

    for i, points in enumerate(landmarks):
        def p(index):
            return points[index]

        bio[i, 0] = calculate_angle(p(LEFT_SHOULDER), p(LEFT_ELBOW), p(LEFT_WRIST))
        bio[i, 1] = calculate_angle(p(RIGHT_SHOULDER), p(RIGHT_ELBOW), p(RIGHT_WRIST))
        bio[i, 2] = calculate_angle(p(LEFT_ELBOW), p(LEFT_SHOULDER), p(LEFT_HIP))
        bio[i, 3] = calculate_angle(p(RIGHT_ELBOW), p(RIGHT_SHOULDER), p(RIGHT_HIP))
        bio[i, 4] = calculate_angle(p(LEFT_SHOULDER), p(LEFT_HIP), p(LEFT_KNEE))
        bio[i, 5] = calculate_angle(p(RIGHT_SHOULDER), p(RIGHT_HIP), p(RIGHT_KNEE))
        bio[i, 6] = calculate_angle(p(LEFT_HIP), p(LEFT_KNEE), p(LEFT_ANKLE))
        bio[i, 7] = calculate_angle(p(RIGHT_HIP), p(RIGHT_KNEE), p(RIGHT_ANKLE))

        shoulder_mid = (p(LEFT_SHOULDER) + p(RIGHT_SHOULDER)) / 2.0
        hip_mid = (p(LEFT_HIP) + p(RIGHT_HIP)) / 2.0
        torso = shoulder_mid - hip_mid
        # Proxies only: axes depend on MediaPipe/camera orientation.
        bio[i, 8] = np.degrees(np.arctan2(torso[2], max(abs(torso[1]), EPSILON)))
        bio[i, 9] = np.degrees(np.arctan2(torso[0], max(abs(torso[1]), EPSILON)))

    # Angular change per resampled frame; first frame has no prior frame.
    bio[0, 10:12] = 0.0
    bio[1:, 10] = np.diff(bio[:, 0])
    bio[1:, 11] = np.diff(bio[:, 1])
    return np.nan_to_num(bio, nan=0.0, posinf=0.0, neginf=0.0)


def augment_sequence(sequence):
    """Append 12 biomechanical features: (30, 99) -> (30, 111)."""
    sequence = np.asarray(sequence, dtype=np.float32)
    biomechanical = calculate_biomechanical_features(sequence)
    augmented = np.concatenate([sequence, biomechanical], axis=1)
    if augmented.shape != (NUM_FRAMES, NUM_FEATURES):
        raise ValueError(f"Unexpected augmented shape: {augmented.shape}")
    if not np.isfinite(augmented).all():
        raise ValueError("Augmented sequence contains non-finite values.")
    return augmented


def load_dataset():
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
                pose_sequence = np.asarray(
                    np.load(file_path, allow_pickle=False), dtype=np.float32
                )
                if pose_sequence.shape != (NUM_FRAMES, NUM_POSE_FEATURES):
                    raise ValueError(
                        f"Expected {(NUM_FRAMES, NUM_POSE_FEATURES)}, "
                        f"got {pose_sequence.shape}. Resample/extract this sequence first."
                    )
                if not np.isfinite(pose_sequence).all():
                    raise ValueError("Sequence contains NaN or infinite values.")
                X.append(augment_sequence(pose_sequence))
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
    print("\n================ E07 DATASET SUMMARY ================")
    for name in CLASSES:
        print(f"{name:>8}: {class_counts[name]} sequences")
    print(f"Total usable sequences: {len(X)}")
    print(f"X shape: {X.shape}")
    print(f"Features per frame: {NUM_FEATURES} (99 pose + 12 biomechanical)")
    print(f"Skipped files: {len(skipped_files)}")

    missing = [name for name, count in class_counts.items() if count < 3]
    if missing:
        raise RuntimeError(f"Need at least 3 valid sequences per class: {missing}")
    return X, y, source_files


def build_model():
    """Retain E06 architecture while adapting input to 111 features."""
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

    print("\n================ E07 DATA SPLIT ================")
    print(f"Training: {len(X_train)}")
    print(f"Validation: {len(X_val)}")
    print(f"Testing: {len(X_test)}")
    print(f"Class order: {CLASSES}")

    # Fit normalization on training split only, across all 111 features.
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
        num_frames=np.asarray(NUM_FRAMES),
        num_pose_features=np.asarray(NUM_POSE_FEATURES),
        num_biomechanical_features=np.asarray(NUM_BIOMECHANICAL_FEATURES),
        feature_version=np.asarray("E07_pose_plus_biomechanical"),
    )
    print(f"Saved E07 normalization stats: {NORMALISATION_PATH}")

    manifest_path = EVALUATION_DIR / "data_split_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        f.write("file,split,class\n")
        for split_name, file_list, labels in [
            ("train", files_train, y_train),
            ("validation", files_val, y_val),
            ("test", files_test, y_test),
        ]:
            for filename, label in zip(file_list, labels):
                f.write(f'"{filename}",{split_name},{CLASSES[int(label)]}\n')

    config = {
        "experiment": "E07_pose_plus_biomechanical",
        "seed": SEED,
        "classes": CLASSES,
        "num_frames": NUM_FRAMES,
        "pose_features_per_frame": NUM_POSE_FEATURES,
        "biomechanical_features_per_frame": NUM_BIOMECHANICAL_FEATURES,
        "total_features_per_frame": NUM_FEATURES,
        "test_size": TEST_SIZE,
        "validation_fraction_of_remainder": VALIDATION_FRACTION_OF_REMAINDER,
        "batch_size": BATCH_SIZE,
        "max_epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "early_stopping_patience": PATIENCE,
        "model_path": str(MODEL_PATH.relative_to(PROJECT_ROOT)),
        "best_model_path": str(BEST_MODEL_PATH.relative_to(PROJECT_ROOT)),
        "normalisation_path": str(NORMALISATION_PATH.relative_to(PROJECT_ROOT)),
        "biomechanical_features": [
            "left_elbow_angle", "right_elbow_angle",
            "left_shoulder_angle", "right_shoulder_angle",
            "left_hip_angle", "right_hip_angle",
            "left_knee_angle", "right_knee_angle",
            "torso_forward_depth_lean_proxy", "torso_lateral_lean_proxy",
            "left_elbow_angular_change", "right_elbow_angular_change",
        ],
    }
    CONFIG_PATH.write_text(json.dumps(config, indent=2), encoding="utf-8")

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

    print("E07 TRAINING STARTED")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
        shuffle=True,
    )

    model.save(MODEL_PATH)
    print(f"Saved E07 model: {MODEL_PATH}")
    print(f"Saved E07 best checkpoint: {BEST_MODEL_PATH}")

    save_training_plot(
        history, "accuracy", "val_accuracy",
        "E07 Training and Validation Accuracy", "Accuracy",
        EVALUATION_DIR / "training_accuracy.png",
    )
    save_training_plot(
        history, "loss", "val_loss",
        "E07 Training and Validation Loss", "Loss",
        EVALUATION_DIR / "training_loss.png",
    )

    test_loss, test_accuracy = model.evaluate(
        X_test, y_test, batch_size=BATCH_SIZE, verbose=0
    )
    probabilities = model.predict(X_test, batch_size=BATCH_SIZE, verbose=0)
    y_pred = np.argmax(probabilities, axis=1)
    report = classification_report(
        y_test, y_pred, labels=list(range(len(CLASSES))),
        target_names=CLASSES, digits=4, zero_division=0,
    )
    cm = confusion_matrix(y_test, y_pred, labels=list(range(len(CLASSES))))
    print(f"\nE07 test loss: {test_loss:.4f}")
    print(f"E07 test accuracy: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")
    print("\nE07 classification report:\n" + report)
    print("Confusion matrix (rows=actual, columns=predicted):")
    print(cm)

    report_path = EVALUATION_DIR / "classification_report.txt"
    report_path.write_text(
        f"Experiment: E07 pose + biomechanical features\n"
        f"Test loss: {test_loss:.6f}\nTest accuracy: {test_accuracy:.6f}\n"
        f"Class order: {', '.join(CLASSES)}\n"
        f"Input shape per sequence: ({NUM_FRAMES}, {NUM_FEATURES})\n\n"
        f"{report}\nConfusion matrix (rows=actual, columns=predicted):\n"
        f"{np.array2string(cm)}",
        encoding="utf-8",
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=[name.title() for name in CLASSES]
    )
    display.plot(values_format="d")
    plt.title("E07 Test Set Confusion Matrix")
    plt.tight_layout()
    plt.savefig(EVALUATION_DIR / "confusion_matrix.png", dpi=150)
    plt.close()

    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, labels=list(range(len(CLASSES))), zero_division=0
    )
    metrics_path = EVALUATION_DIR / "per_class_metrics.csv"
    with metrics_path.open("w", encoding="utf-8", newline="") as f:
        f.write("class,precision,recall,f1_score,support\n")
        for i, name in enumerate(CLASSES):
            f.write(
                f"{name},{precision[i]:.6f},{recall[i]:.6f},"
                f"{f1[i]:.6f},{int(support[i])}\n"
            )

    print("\n================ E07 OUTPUTS ================")
    for output in [
        MODEL_PATH, BEST_MODEL_PATH, NORMALISATION_PATH, CONFIG_PATH,
        manifest_path, EVALUATION_DIR / "training_accuracy.png",
        EVALUATION_DIR / "training_loss.png", report_path,
        EVALUATION_DIR / "confusion_matrix.png", metrics_path,
    ]:
        print(output)
    print(
        "\nE07 training finished. E06 files were not overwritten. "
        "The backend must calculate the same 12 biomechanical features "
        "and load the E07 model plus E07 normalization stats before E07 inference."
    )


if __name__ == "__main__":
    main()

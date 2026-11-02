
import os
import random
import json
import csv
import re
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# ============================================================
# 700 PER CLASS: FULL-FEATURE LSTM CLASSIFIER
# ============================================================

SEED = 42
CLASSES = ["clear", "drop", "smash"]

NUM_FRAMES = 30

# Use exactly 700 valid pose sequences from each stroke class.
MAX_SAMPLES_PER_CLASS = 700

NUM_POSE_FEATURES = 99
NUM_BIOMECHANICAL_FEATURES = 12
NUM_ANGULAR_VELOCITY_FEATURES = 8
NUM_ANGULAR_ACCELERATION_FEATURES = 8
NUM_RELATIVE_TIMING_FEATURES = 12

NUM_TEMPORAL_FEATURES = (
    NUM_ANGULAR_VELOCITY_FEATURES
    + NUM_ANGULAR_ACCELERATION_FEATURES
    + NUM_RELATIVE_TIMING_FEATURES
)

NUM_FEATURES = (
    NUM_POSE_FEATURES
    + NUM_BIOMECHANICAL_FEATURES
    + NUM_TEMPORAL_FEATURES
)

TEST_SIZE = 0.15
VALIDATION_SIZE = 0.15

# TUNED TRAINING CONFIGURATION
BATCH_SIZE = 16
EPOCHS = 150
LEARNING_RATE = 0.0001
PATIENCE = 12

LSTM_UNITS = 32
DENSE_UNITS = 16
DROPOUT_RATE = 0.35
L2_RATE = 0.0005
LR_REDUCTION_PATIENCE = 4

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SEQUENCES_DIR = PROJECT_ROOT / "dataset" / "sequences"
MODELS_DIR = PROJECT_ROOT / "models"

EVALUATION_BASE_DIR = (
    PROJECT_ROOT / "evaluation" / "700_per_class_full_features"
)

# These paths are assigned to a unique run directory at the start of main().
EVALUATION_DIR = None
RUN_ID = None
RUN_TIMESTAMP = None

MODEL_PATH = None
BEST_MODEL_PATH = None
NORMALISATION_PATH = None
CONFIG_PATH = None

EPSILON = 1e-7

# MediaPipe landmark indices
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_ELBOW, RIGHT_ELBOW = 13, 14
LEFT_WRIST, RIGHT_WRIST = 15, 16
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_KNEE, RIGHT_KNEE = 25, 26
LEFT_ANKLE, RIGHT_ANKLE = 27, 28


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seeds():

    os.environ.setdefault("PYTHONHASHSEED", str(SEED))

    random.seed(SEED)
    np.random.seed(SEED)

    tf.random.set_seed(SEED)


# ============================================================
# COMBINE POSE + BIOMECHANICAL + TEMPORAL FEATURES
# ============================================================

# ============================================================
# BIOMECHANICAL AND TEMPORAL FEATURE EXTRACTION
# ============================================================

def calculate_angle(a, b, c):
    """Calculate angle ABC in degrees using 3D coordinates."""

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

    return float(
        np.degrees(
            np.arccos(np.clip(cosine, -1.0, 1.0))
        )
    )


# ============================================================
# BIOMECHANICAL FEATURE EXTRACTION
# ============================================================

def calculate_biomechanical_features(sequence):
    """
    Calculate 12 biomechanical features:

    0-7: 8 joint angles
    8-9: 2 torso lean proxies
    10-11: Left and right elbow-angle changes
    """

    sequence = np.asarray(sequence, dtype=np.float32)

    if sequence.shape != (NUM_FRAMES, NUM_POSE_FEATURES):
        raise ValueError(
            f"Expected {(NUM_FRAMES, NUM_POSE_FEATURES)}, "
            f"got {sequence.shape}"
        )

    landmarks = sequence.reshape(NUM_FRAMES, 33, 3)

    bio = np.zeros(
        (NUM_FRAMES, NUM_BIOMECHANICAL_FEATURES),
        dtype=np.float32
    )

    for i, points in enumerate(landmarks):

        def p(index):
            return points[index]

        # Elbow angles
        bio[i, 0] = calculate_angle(
            p(LEFT_SHOULDER),
            p(LEFT_ELBOW),
            p(LEFT_WRIST)
        )

        bio[i, 1] = calculate_angle(
            p(RIGHT_SHOULDER),
            p(RIGHT_ELBOW),
            p(RIGHT_WRIST)
        )

        # Shoulder angles
        bio[i, 2] = calculate_angle(
            p(LEFT_ELBOW),
            p(LEFT_SHOULDER),
            p(LEFT_HIP)
        )

        bio[i, 3] = calculate_angle(
            p(RIGHT_ELBOW),
            p(RIGHT_SHOULDER),
            p(RIGHT_HIP)
        )

        # Hip angles
        bio[i, 4] = calculate_angle(
            p(LEFT_SHOULDER),
            p(LEFT_HIP),
            p(LEFT_KNEE)
        )

        bio[i, 5] = calculate_angle(
            p(RIGHT_SHOULDER),
            p(RIGHT_HIP),
            p(RIGHT_KNEE)
        )

        # Knee angles
        bio[i, 6] = calculate_angle(
            p(LEFT_HIP),
            p(LEFT_KNEE),
            p(LEFT_ANKLE)
        )

        bio[i, 7] = calculate_angle(
            p(RIGHT_HIP),
            p(RIGHT_KNEE),
            p(RIGHT_ANKLE)
        )

        # Torso lean proxies
        shoulder_mid = (
            p(LEFT_SHOULDER) + p(RIGHT_SHOULDER)
        ) / 2.0

        hip_mid = (
            p(LEFT_HIP) + p(RIGHT_HIP)
        ) / 2.0

        torso = shoulder_mid - hip_mid

        bio[i, 8] = np.degrees(
            np.arctan2(
                torso[2],
                max(abs(torso[1]), EPSILON)
            )
        )

        bio[i, 9] = np.degrees(
            np.arctan2(
                torso[0],
                max(abs(torso[1]), EPSILON)
            )
        )

    # Elbow-angle changes between consecutive frames
    bio[0, 10:12] = 0.0

    bio[1:, 10] = np.diff(bio[:, 0])
    bio[1:, 11] = np.diff(bio[:, 1])

    return np.nan_to_num(
        bio,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )


# ============================================================
# TEMPORAL FEATURE EXTRACTION
# ============================================================

def calculate_temporal_features(biomechanical):
    """
    Generate 28 temporal features per frame:

    0-7: 8 angular velocities
    8-15: 8 angular accelerations
    16-23: Peak timing of 8 joint angles
    24-27: Peak timing of 4 angular velocities

    Derivatives use normalized sequence time [0, 1].
    """

    angles = np.asarray(
        biomechanical[:, :8],
        dtype=np.float32
    )

    time_axis = np.linspace(
        0.0,
        1.0,
        NUM_FRAMES,
        dtype=np.float32
    )

    # Angular velocity
    velocity = np.gradient(
        angles,
        time_axis,
        axis=0
    ).astype(np.float32)

    # Angular acceleration
    acceleration = np.gradient(
        velocity,
        time_axis,
        axis=0
    ).astype(np.float32)

    # Peak timing of all 8 joint angles
    angle_peak_timing = np.zeros(
        8,
        dtype=np.float32
    )

    for j in range(8):

        angle_peak_timing[j] = time_axis[
            int(np.argmax(angles[:, j]))
        ]

    # Peak absolute angular velocity timing
    # Left/right elbow and left/right shoulder
    velocity_joint_indices = [0, 1, 2, 3]

    velocity_peak_timing = np.zeros(
        4,
        dtype=np.float32
    )

    for out_idx, joint_idx in enumerate(
        velocity_joint_indices
    ):

        velocity_peak_timing[out_idx] = time_axis[
            int(np.argmax(np.abs(velocity[:, joint_idx])))
        ]

    # Combine 8 angle timings + 4 velocity timings
    timing = np.concatenate(
        [
            angle_peak_timing,
            velocity_peak_timing
        ]
    ).astype(np.float32)

    # Repeat sequence-level timing descriptors across frames
    timing_per_frame = np.repeat(
        timing[None, :],
        NUM_FRAMES,
        axis=0
    )

    # Final temporal feature vector:
    # 8 velocity + 8 acceleration + 12 timing = 28
    temporal = np.concatenate(
        [
            velocity,
            acceleration,
            timing_per_frame
        ],
        axis=1
    )

    if temporal.shape != (
        NUM_FRAMES,
        NUM_TEMPORAL_FEATURES
    ):
        raise ValueError(
            f"Unexpected temporal shape: {temporal.shape}"
        )

    return np.nan_to_num(
        temporal,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )




# ============================================================
# COMBINE POSE + BIOMECHANICAL + TEMPORAL FEATURES
# ============================================================

def augment_sequence(sequence):
    """
    Full model input:
      99 pose coordinates
      12 biomechanical features
      28 temporal features
      Total: 139 features per frame.
    """
    sequence = np.asarray(sequence, dtype=np.float32)

    if sequence.shape != (NUM_FRAMES, NUM_POSE_FEATURES):
        raise ValueError(
            f"Expected {(NUM_FRAMES, NUM_POSE_FEATURES)}, got {sequence.shape}"
        )

    if not np.isfinite(sequence).all():
        raise ValueError("Pose sequence contains non-finite values.")

    biomechanical = calculate_biomechanical_features(sequence)
    temporal = calculate_temporal_features(biomechanical)

    augmented = np.concatenate(
        [sequence, biomechanical, temporal],
        axis=1
    )

    if augmented.shape != (NUM_FRAMES, NUM_FEATURES):
        raise ValueError(
            f"Unexpected feature shape: {augmented.shape}"
        )

    if not np.isfinite(augmented).all():
        raise ValueError("Augmented sequence contains non-finite values.")

    return augmented


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset():

    X, y, source_files = [], [], []

    class_counts = {
        name: 0 for name in CLASSES
    }

    skipped_files = []

    for label, class_name in enumerate(CLASSES):

        class_dir = SEQUENCES_DIR / class_name

        if not class_dir.exists():
            raise FileNotFoundError(
                f"Missing class folder: {class_dir}"
            )

        for file_path in sorted(
            class_dir.glob("*.npy")
        ):
            # Stop after collecting 700 valid sequences for this class.
            if class_counts[class_name] >= MAX_SAMPLES_PER_CLASS:
                break

            try:

                pose_sequence = np.asarray(
                    np.load(
                        file_path,
                        allow_pickle=False
                    ),
                    dtype=np.float32
                )

                if pose_sequence.shape != (
                    NUM_FRAMES,
                    NUM_POSE_FEATURES
                ):

                    raise ValueError(
                        f"Expected "
                        f"{(NUM_FRAMES, NUM_POSE_FEATURES)}, "
                        f"got {pose_sequence.shape}"
                    )

                if not np.isfinite(pose_sequence).all():

                    raise ValueError(
                        "Sequence contains NaN or infinite values."
                    )

                X.append(
                    augment_sequence(pose_sequence)
                )

                y.append(label)

                source_files.append(
                    str(
                        file_path.relative_to(PROJECT_ROOT)
                    )
                )

                class_counts[class_name] += 1

            except Exception as exc:

                skipped_files.append(
                    (file_path, str(exc))
                )

                print(
                    f"SKIPPED: {file_path.name} - {exc}"
                )

    if not X:
        raise RuntimeError(
            f"No valid sequences found in {SEQUENCES_DIR}"
        )

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int32)

    print(
        "\n================ 700 PER CLASS FULL-FEATURE DATASET SUMMARY ================"
    )

    for name in CLASSES:
        print(
            f"{name:>8}: {class_counts[name]} sequences"
        )

    print(f"Total usable sequences: {len(X)}")
    print(f"Dataset shape: {X.shape}")
    print(f"Features per frame: {NUM_FEATURES}")
    print("Feature composition: 99 pose + 12 biomechanical + 28 temporal features")
    print(f"Skipped files: {len(skipped_files)}")
    print(f"Target per class: {MAX_SAMPLES_PER_CLASS}")

    insufficient = {
        name: count for name, count in class_counts.items()
        if count < MAX_SAMPLES_PER_CLASS
    }
    if insufficient:
        details = ", ".join(
            f"{name}={count}/{MAX_SAMPLES_PER_CLASS}"
            for name, count in insufficient.items()
        )
        raise RuntimeError(
            "Not enough valid pose sequences for the requested balanced "
            f"dataset: {details}. Extract more videos before training."
        )

    if any(
        count < 3
        for count in class_counts.values()
    ):
        raise RuntimeError(
            "Need at least 3 valid sequences per class."
        )

    return X, y, source_files


# ============================================================
# BUILD LSTM MODEL
# ============================================================

def build_model():

    regularizer = tf.keras.regularizers.l2(L2_RATE)

    model = tf.keras.Sequential([

        tf.keras.layers.Input(
            shape=(NUM_FRAMES, NUM_FEATURES)
        ),

        tf.keras.layers.LSTM(
            LSTM_UNITS,
            kernel_regularizer=regularizer,
            recurrent_regularizer=regularizer
        ),

        tf.keras.layers.Dropout(DROPOUT_RATE),

        tf.keras.layers.Dense(
            DENSE_UNITS,
            activation="relu",
            kernel_regularizer=regularizer
        ),

        tf.keras.layers.Dense(
            len(CLASSES),
            activation="softmax"
        )
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ============================================================
# SAVE TRAINING GRAPHS
# ============================================================

def save_training_plot(
    history,
    key,
    val_key,
    title,
    ylabel,
    output_path
):

    epochs = range(
        1,
        len(history.history[key]) + 1
    )

    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        history.history[key],
        label=f"Training {ylabel.lower()}"
    )

    plt.plot(
        epochs,
        history.history[val_key],
        label=f"Validation {ylabel.lower()}"
    )

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=150
    )

    plt.close()


# ============================================================
# EXPERIMENT RUN VERSIONING
# ============================================================

def create_run_directory():
    """
    Create a new run_NNN folder without overwriting previous runs.
    All artifacts for this training run are stored inside that folder.
    """
    EVALUATION_BASE_DIR.mkdir(parents=True, exist_ok=True)

    existing_numbers = []
    for path in EVALUATION_BASE_DIR.glob("run_*"):
        if path.is_dir():
            match = re.fullmatch(r"run_(\d+)", path.name)
            if match:
                existing_numbers.append(int(match.group(1)))

    next_number = max(existing_numbers, default=0) + 1
    run_id = f"run_{next_number:03d}"
    run_dir = EVALUATION_BASE_DIR / run_id

    # exist_ok=False protects against accidentally reusing a run folder.
    run_dir.mkdir(parents=False, exist_ok=False)

    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    return run_id, timestamp, run_dir


# ============================================================
# MAIN TRAINING PROCESS
# ============================================================

def main():
    global EVALUATION_DIR
    global MODEL_PATH, BEST_MODEL_PATH, NORMALISATION_PATH, CONFIG_PATH
    global RUN_ID, RUN_TIMESTAMP

    set_seeds()

    RUN_ID, RUN_TIMESTAMP, EVALUATION_DIR = create_run_directory()

    # Keep each run completely self-contained: models, normalization,
    # configuration, split manifest, metrics, and plots are all versioned.
    MODEL_PATH = EVALUATION_DIR / "final_model.keras"
    BEST_MODEL_PATH = EVALUATION_DIR / "best_model.keras"
    NORMALISATION_PATH = EVALUATION_DIR / "normalisation_stats.npz"
    CONFIG_PATH = EVALUATION_DIR / "training_config.json"

    print(f"\\nCreated experiment run: {RUN_ID}")
    print(f"Run output directory: {EVALUATION_DIR}")

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    X, y, source_files = load_dataset()

    # --------------------------------------------------------
    # TRAINING / VALIDATION / TEST SPLIT 
    # --------------------------------------------------------

    (
        X_train,
        X_temp,
        y_train,
        y_temp,
        files_train,
        files_temp
    ) = train_test_split(
        X,
        y,
        source_files,
        test_size=(VALIDATION_SIZE + TEST_SIZE),
        random_state=SEED,
        stratify=y
    )

    # Split the remaining 30% equally into validation and test.
    relative_test_size = TEST_SIZE / (VALIDATION_SIZE + TEST_SIZE)

    (
        X_val,
        X_test,
        y_val,
        y_test,
        files_val,
        files_test
    ) = train_test_split(
        X_temp,
        y_temp,
        files_temp,
        test_size=relative_test_size,
        random_state=SEED,
        stratify=y_temp
    )

    print(
        "\\n================ 700 PER CLASS FULL-FEATURE DATA SPLIT ================"
    )

    print(f"Total dataset: {len(X)} (700 per class)")
    print(f"Training: {len(X_train)} ({len(X_train) / len(X):.1%})")
    print(f"Validation: {len(X_val)} ({len(X_val) / len(X):.1%})")
    print(f"Testing: {len(X_test)} ({len(X_test) / len(X):.1%})")
    print(f"Class order: {CLASSES}")

    # --------------------------------------------------------
    # NORMALIZATION FITTED ON TRAINING DATA ONLY
    # --------------------------------------------------------

    mean = np.mean(
        X_train,
        axis=(0, 1),
        keepdims=True
    ).astype(np.float32)

    std = np.std(
        X_train,
        axis=(0, 1),
        keepdims=True
    ).astype(np.float32)

    std[std < 1e-7] = 1.0

    X_train = (
        (X_train - mean) / std
    ).astype(np.float32)

    X_val = (
        (X_val - mean) / std
    ).astype(np.float32)

    # Apply the training-set statistics to the untouched test set.
    X_test = (
        (X_test - mean) / std
    ).astype(np.float32)

    # --------------------------------------------------------
    # SAVE NORMALIZATION STATISTICS
    # --------------------------------------------------------

    np.savez(
        NORMALISATION_PATH,
        mean=mean.reshape(NUM_FEATURES),
        std=std.reshape(NUM_FEATURES),
        classes=np.asarray(CLASSES),
        num_frames=np.asarray(NUM_FRAMES),
        num_pose_features=np.asarray(NUM_POSE_FEATURES),
        num_biomechanical_features=np.asarray(NUM_BIOMECHANICAL_FEATURES),
        num_temporal_features=np.asarray(NUM_TEMPORAL_FEATURES),
        feature_version=np.asarray("Full_Features_700_per_class")
    )

    print(
        f"Saved normalization statistics: {NORMALISATION_PATH}"
    )

    # --------------------------------------------------------
    # SAVE TRAINING / VALIDATION MANIFEST
    # --------------------------------------------------------

    manifest_path = (
        EVALUATION_DIR / "data_split_manifest.csv"
    )

    with manifest_path.open(
        "w",
        encoding="utf-8",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "file",
            "split",
            "class"
        ])

        for split_name, file_list, labels in [
            ("train", files_train, y_train),
            ("validation", files_val, y_val),
            ("test", files_test, y_test)
        ]:

            for filename, label in zip(
                file_list,
                labels
            ):

                writer.writerow([
                    filename,
                    split_name,
                    CLASSES[int(label)]
                ])

    # --------------------------------------------------------
    # SAVE EXPERIMENT CONFIGURATION
    # --------------------------------------------------------

    config = {
        "experiment": "700_per_class_full_features",
        "run_id": RUN_ID,
        "run_timestamp": RUN_TIMESTAMP,
        "samples_per_class": MAX_SAMPLES_PER_CLASS,
        "total_samples": int(len(X)),
        "dataset_directory": str(SEQUENCES_DIR.relative_to(PROJECT_ROOT)),
        "seed": SEED,
        "classes": CLASSES,
        "num_frames": NUM_FRAMES,

        "pose_features_per_frame": NUM_POSE_FEATURES,
        "biomechanical_features_per_frame": NUM_BIOMECHANICAL_FEATURES,
        "temporal_features_per_frame": NUM_TEMPORAL_FEATURES,
        "total_features_per_frame": NUM_FEATURES,

        "feature_composition": "99 pose + 12 biomechanical + 28 temporal features",

        "training_size": 0.70,
        "validation_size": VALIDATION_SIZE,
        "test_size": TEST_SIZE,
        "split_strategy": "stratified_train_validation_test",
        "internal_test_split": True,

        "batch_size": BATCH_SIZE,
        "max_epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "early_stopping_patience": PATIENCE,
        "lr_reduction_patience": LR_REDUCTION_PATIENCE,

        "lstm_units": LSTM_UNITS,
        "dense_units": DENSE_UNITS,
        "dropout": DROPOUT_RATE,
        "l2_regularization": L2_RATE,

        "model_path": str(
            MODEL_PATH.relative_to(PROJECT_ROOT)
        ),

        "best_model_path": str(
            BEST_MODEL_PATH.relative_to(PROJECT_ROOT)
        ),

        "normalisation_path": str(
            NORMALISATION_PATH.relative_to(PROJECT_ROOT)
        )
    }

    CONFIG_PATH.write_text(
        json.dumps(config, indent=2),
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # BUILD MODEL
    # --------------------------------------------------------

    model = build_model()

    model.summary()

    # --------------------------------------------------------
    # TRAINING CALLBACKS
    # --------------------------------------------------------

    callbacks = [

        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),

        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(BEST_MODEL_PATH),
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=LR_REDUCTION_PATIENCE,
            min_lr=1e-6,
            verbose=1
        )
    ]

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    print(
        "\n================ 700 PER CLASS FULL-FEATURE TRAINING STARTED ================"
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
        shuffle=True
    )

    # Save per-epoch training history for later comparison.
    history_path = EVALUATION_DIR / "training_history.csv"
    history_keys = list(history.history.keys())
    with history_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch"] + history_keys)
        for epoch_index in range(len(history.history[history_keys[0]])):
            writer.writerow(
                [epoch_index + 1]
                + [history.history[key][epoch_index] for key in history_keys]
            )

    # Save final model with restored best weights
    model.save(MODEL_PATH)

    print(f"\nSaved final model: {MODEL_PATH}")
    print(f"Saved best checkpoint: {BEST_MODEL_PATH}")

    # --------------------------------------------------------
    # FINAL TEST-SET EVALUATION
    # --------------------------------------------------------

    print("\\n================ FINAL TEST-SET EVALUATION ================")

    test_probabilities = model.predict(X_test, verbose=0)
    y_pred = np.argmax(test_probabilities, axis=1)

    test_accuracy = accuracy_score(y_test, y_pred)
    test_report = classification_report(
        y_test, y_pred,
        labels=list(range(len(CLASSES))),
        target_names=CLASSES,
        output_dict=True,
        zero_division=0
    )
    test_report_text = classification_report(
        y_test, y_pred,
        labels=list(range(len(CLASSES))),
        target_names=CLASSES,
        zero_division=0
    )
    test_confusion = confusion_matrix(
        y_test, y_pred,
        labels=list(range(len(CLASSES)))
    )

    print(f"Test accuracy: {test_accuracy:.4f} ({test_accuracy:.2%})")
    print("\\nClassification report:")
    print(test_report_text)
    print("Confusion matrix (rows=actual, columns=predicted):")
    print(test_confusion)

    (EVALUATION_DIR / "test_classification_report.txt").write_text(
        test_report_text, encoding="utf-8"
    )
    (EVALUATION_DIR / "test_metrics.json").write_text(
        json.dumps({
            "test_accuracy": float(test_accuracy),
            "classification_report": test_report,
            "class_order": CLASSES,
            "confusion_matrix": test_confusion.tolist(),
            "test_samples": int(len(y_test))
        }, indent=2),
        encoding="utf-8"
    )
    np.savetxt(
        EVALUATION_DIR / "test_confusion_matrix.csv",
        test_confusion,
        delimiter=",",
        fmt="%d"
    )

    # --------------------------------------------------------
    # SAVE ACCURACY GRAPH
    # --------------------------------------------------------

    save_training_plot(
        history,
        "accuracy",
        "val_accuracy",
        "Full-Feature LSTM (700 per class) Training and Validation Accuracy",
        "Accuracy",
        EVALUATION_DIR / "training_accuracy.png"
    )

    # --------------------------------------------------------
    # SAVE LOSS GRAPH
    # --------------------------------------------------------

    save_training_plot(
        history,
        "loss",
        "val_loss",
        "Full-Feature LSTM (700 per class) Training and Validation Loss",
        "Loss",
        EVALUATION_DIR / "training_loss.png"
    )

    # --------------------------------------------------------
    # OUTPUT SUMMARY
    # --------------------------------------------------------

    print("\n================ 700 PER CLASS FULL-FEATURE OUTPUTS ================")

    for output in [
        MODEL_PATH,
        BEST_MODEL_PATH,
        NORMALISATION_PATH,
        CONFIG_PATH,
        manifest_path,
        EVALUATION_DIR / "training_accuracy.png",
        EVALUATION_DIR / "training_loss.png",
        EVALUATION_DIR / "training_history.csv",
        EVALUATION_DIR / "test_classification_report.txt",
        EVALUATION_DIR / "test_metrics.json",
        EVALUATION_DIR / "test_confusion_matrix.csv"
    ]:

        print(output)

    print(
        "\nFULL-FEATURE LSTM TRAINING WITH 700 VIDEOS PER CLASS COMPLETED."
    )


if __name__ == "__main__":
    main()

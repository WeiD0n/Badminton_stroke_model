
import os
import random
import json
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# ============================================================
# 700 PER CLASS: POSE-ONLY LSTM CLASSIFIER
# ============================================================

SEED = 42
CLASSES = ["clear", "drop", "smash"]

NUM_FRAMES = 30

# Use exactly 700 valid pose sequences from each stroke class.
MAX_SAMPLES_PER_CLASS = 700

NUM_POSE_FEATURES = 99
NUM_FEATURES = NUM_POSE_FEATURES

TEST_SIZE = 0.15
VALIDATION_SIZE = 0.15

# TUNED TRAINING CONFIGURATION
BATCH_SIZE = 16
EPOCHS = 150
LEARNING_RATE = 0.0005
PATIENCE = 12

LSTM_UNITS = 32
DENSE_UNITS = 16
DROPOUT_RATE = 0.35
L2_RATE = 0.0005
LR_REDUCTION_PATIENCE = 4

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SEQUENCES_DIR = PROJECT_ROOT / "dataset" / "sequences"
MODELS_DIR = PROJECT_ROOT / "models"

EVALUATION_DIR = (
    PROJECT_ROOT / "evaluation" / "700_per_class_pose_only"
)

MODEL_PATH = (
    MODELS_DIR /
    "lstm_stroke_classifier_700_pose_only.keras"
)

BEST_MODEL_PATH = (
    MODELS_DIR /
    "lstm_stroke_classifier_700_pose_only_best.keras"
)

NORMALISATION_PATH = (
    MODELS_DIR /
    "normalisation_stats_700_per_class_pose_only.npz"
)

CONFIG_PATH = EVALUATION_DIR / "training_config.json"

EPSILON = 1e-7


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

def augment_sequence(sequence):
    """
    Pose-only input: 33 MediaPipe landmarks x (x, y, z) = 99 features.
    No biomechanical or temporal features are added.
    """
    sequence = np.asarray(sequence, dtype=np.float32)

    if sequence.shape != (NUM_FRAMES, NUM_POSE_FEATURES):
        raise ValueError(
            f"Expected {(NUM_FRAMES, NUM_POSE_FEATURES)}, got {sequence.shape}"
        )

    if not np.isfinite(sequence).all():
        raise ValueError("Pose sequence contains non-finite values.")

    return sequence


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
        "\n================ 700 PER CLASS POSE-ONLY DATASET SUMMARY ================"
    )

    for name in CLASSES:
        print(
            f"{name:>8}: {class_counts[name]} sequences"
        )

    print(f"Total usable sequences: {len(X)}")
    print(f"Dataset shape: {X.shape}")
    print(f"Features per frame: {NUM_FEATURES}")
    print("Feature composition: 99 pose features only")
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
# MAIN TRAINING PROCESS
# ============================================================

def main():

    set_seeds()

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    EVALUATION_DIR.mkdir(
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
        "\\n================ 700 PER CLASS POSE-ONLY DATA SPLIT ================"
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
        num_biomechanical_features=np.asarray(0),
        num_temporal_features=np.asarray(0),
        feature_version=np.asarray("Pose_only_700_per_class")
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
        "experiment": "700_per_class_pose_only",
        "samples_per_class": MAX_SAMPLES_PER_CLASS,
        "total_samples": int(len(X)),
        "dataset_directory": str(SEQUENCES_DIR.relative_to(PROJECT_ROOT)),
        "seed": SEED,
        "classes": CLASSES,
        "num_frames": NUM_FRAMES,

        "pose_features_per_frame": NUM_POSE_FEATURES,
        "biomechanical_features_per_frame": 0,
        "temporal_features_per_frame": 0,
        "total_features_per_frame": NUM_FEATURES,

        "feature_composition": "99 pose features only",

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
        "\n================ 700 PER CLASS POSE-ONLY TRAINING STARTED ================"
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
        "Pose-Only LSTM (700 per class) Training and Validation Accuracy",
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
        "Pose-Only LSTM (700 per class) Training and Validation Loss",
        "Loss",
        EVALUATION_DIR / "training_loss.png"
    )

    # --------------------------------------------------------
    # OUTPUT SUMMARY
    # --------------------------------------------------------

    print("\n================ 700 PER CLASS POSE-ONLY OUTPUTS ================")

    for output in [
        MODEL_PATH,
        BEST_MODEL_PATH,
        NORMALISATION_PATH,
        CONFIG_PATH,
        manifest_path,
        EVALUATION_DIR / "training_accuracy.png",
        EVALUATION_DIR / "training_loss.png",
        EVALUATION_DIR / "test_classification_report.txt",
        EVALUATION_DIR / "test_metrics.json",
        EVALUATION_DIR / "test_confusion_matrix.csv"
    ]:

        print(output)

    print(
        "\nPOSE-ONLY LSTM TRAINING WITH 700 VIDEOS PER CLASS COMPLETED."
    )


if __name__ == "__main__":
    main()

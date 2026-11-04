
import os
import json
import random
import csv
from pathlib import Path

import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from training.train_lstm_full_feature import augment_sequence


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

CLASSES = ["clear", "drop", "smash"]

NUM_FRAMES = 30
NUM_FEATURES = 139

EXPECTED_SAMPLES_PER_CLASS = 490

# Internal split of the 490 training sequences.
VALIDATION_SIZE = 0.20

BATCH_SIZE = 16
EPOCHS = 100
PATIENCE = 12
LEARNING_RATE = 0.0001

LSTM_UNITS = 128
LATENT_UNITS = 64
DROPOUT_RATE = 0.20

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RUN_DIR = (
    PROJECT_ROOT
    / "evaluation"
    / "700_per_class_full_features"
    / "run_059"
)

MANIFEST_PATH = RUN_DIR / "data_split_manifest.csv"

NORMALISATION_PATH = (
    RUN_DIR / "normalisation_stats.npz"
)

SEQUENCES_DIR = (
    PROJECT_ROOT / "dataset" / "sequences"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "models"
    / "form_evaluation"
    / "autoencoders"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seeds():

    os.environ["PYTHONHASHSEED"] = str(SEED)

    random.seed(SEED)
    np.random.seed(SEED)

    tf.keras.utils.set_random_seed(SEED)


# ============================================================
# LOAD NORMALIZATION STATISTICS
# ============================================================

def load_normalisation():

    if not NORMALISATION_PATH.exists():
        raise FileNotFoundError(
            f"Normalization file not found:\n"
            f"{NORMALISATION_PATH}"
        )

    data = np.load(
        NORMALISATION_PATH,
        allow_pickle=False
    )

    mean = np.asarray(
        data["mean"],
        dtype=np.float32
    )

    std = np.asarray(
        data["std"],
        dtype=np.float32
    )

    if mean.shape != (NUM_FEATURES,):
        raise ValueError(
            f"Unexpected mean shape: {mean.shape}"
        )

    if std.shape != (NUM_FEATURES,):
        raise ValueError(
            f"Unexpected std shape: {std.shape}"
        )

    std[std < 1e-7] = 1.0

    print("Loaded classifier normalization statistics.")

    return mean, std


# ============================================================
# LOAD TRAINING MANIFEST
# ============================================================

def load_training_manifest():

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Manifest not found:\n{MANIFEST_PATH}"
        )

    training_files = {
        class_name: []
        for class_name in CLASSES
    }

    with open(
        MANIFEST_PATH,
        "r",
        newline="",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)

        required_columns = {
            "file",
            "split",
            "class"
        }

        if not required_columns.issubset(
            reader.fieldnames or []
        ):
            raise ValueError(
                "Invalid manifest columns."
            )

        for row in reader:

            if row["split"].strip() != "train":
                continue

            class_name = row["class"].strip()

            if class_name not in CLASSES:
                raise ValueError(
                    f"Unknown class: {class_name}"
                )

            file_path = (
                PROJECT_ROOT / Path(row["file"])
            )

            training_files[class_name].append(
                file_path
            )

    for class_name in CLASSES:

        count = len(
            training_files[class_name]
        )

        if count != EXPECTED_SAMPLES_PER_CLASS:
            raise ValueError(
                f"{class_name}: expected "
                f"{EXPECTED_SAMPLES_PER_CLASS} sequences, "
                f"found {count}."
            )

    print("\nTraining manifest verified.")

    for class_name in CLASSES:
        print(
            f"{class_name.capitalize()}: "
            f"{len(training_files[class_name])}"
        )

    return training_files


# ============================================================
# LOAD AND NORMALIZE SEQUENCES
# ============================================================

def load_class_sequences(
    class_name,
    file_paths,
    mean,
    std
):

    sequences = []

    print(
        f"\nLoading {class_name} sequences..."
    )

    for index, file_path in enumerate(
        file_paths,
        start=1
    ):

        if not file_path.exists():
            raise FileNotFoundError(
                f"Missing sequence: {file_path}"
            )

        sequence = np.asarray(
            np.load(
                file_path,
                allow_pickle=False
            ),
            dtype=np.float32
        )

        # Original sequences contain 99 pose features.
        if sequence.shape != (NUM_FRAMES, 99):
            raise ValueError(
                f"Invalid pose sequence shape in {file_path}: "
                f"{sequence.shape}"
            )

        # Convert 99 pose features into 139 full features.
        sequence = augment_sequence(sequence)

        if sequence.shape != (
            NUM_FRAMES,
            NUM_FEATURES
        ):
            raise ValueError(
                f"Invalid augmented shape in {file_path}: "
                f"{sequence.shape}"
            )

        if not np.isfinite(sequence).all():
            raise ValueError(
                f"Non-finite values in {file_path}"
            )

        # Use the same normalization statistics
        # as the selected seed 42 classifier.
        sequence = (
            sequence - mean
        ) / std

        sequences.append(sequence)

        if index % 100 == 0:
            print(
                f"Processed {index}/"
                f"{len(file_paths)}"
            )

    X = np.asarray(
        sequences,
        dtype=np.float32
    )

    expected_shape = (
        EXPECTED_SAMPLES_PER_CLASS,
        NUM_FRAMES,
        NUM_FEATURES
    )

    if X.shape != expected_shape:
        raise ValueError(
            f"Unexpected dataset shape: {X.shape}"
        )

    return X


# ============================================================
# BUILD LSTM AUTOENCODER
# ============================================================

def build_autoencoder():

    inputs = tf.keras.Input(
        shape=(NUM_FRAMES, NUM_FEATURES),
        name="input_sequence"
    )

    # Encoder
    encoded = tf.keras.layers.LSTM(
        LSTM_UNITS,
        return_sequences=False,
        name="encoder_lstm"
    )(inputs)

    encoded = tf.keras.layers.Dropout(
        DROPOUT_RATE
    )(encoded)

    latent = tf.keras.layers.Dense(
        LATENT_UNITS,
        activation="tanh",
        name="latent_representation"
    )(encoded)

    # Decoder
    decoded = tf.keras.layers.RepeatVector(
        NUM_FRAMES
    )(latent)

    decoded = tf.keras.layers.LSTM(
        LSTM_UNITS,
        return_sequences=True,
        name="decoder_lstm"
    )(decoded)

    decoded = tf.keras.layers.Dropout(
        DROPOUT_RATE
    )(decoded)

    outputs = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(NUM_FEATURES),
        name="reconstructed_sequence"
    )(decoded)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="badminton_lstm_autoencoder"
    )

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    )

    model.compile(
        optimizer=optimizer,
        loss="mse",
        metrics=["mae"]
    )

    return model


# ============================================================
# CALCULATE RECONSTRUCTION ERRORS
# ============================================================

def calculate_reconstruction_errors(
    model,
    X
):

    reconstructed = model.predict(
        X,
        batch_size=BATCH_SIZE,
        verbose=0
    )

    # One reconstruction error per sequence.
    errors = np.mean(
        np.square(
            X - reconstructed
        ),
        axis=(1, 2)
    )

    return errors.astype(np.float32)


# ============================================================
# TRAIN ONE CLASS-SPECIFIC AUTOENCODER
# ============================================================

def train_class_autoencoder(
    class_name,
    X
):

    print("\n" + "=" * 60)
    print(
        f"TRAINING {class_name.upper()} AUTOENCODER"
    )
    print("=" * 60)

    X_train, X_val = train_test_split(
        X,
        test_size=VALIDATION_SIZE,
        random_state=SEED,
        shuffle=True
    )

    print(
        f"Training sequences: {len(X_train)}"
    )

    print(
        f"Validation sequences: {len(X_val)}"
    )

    model = build_autoencoder()

    model.summary()

    class_output_dir = (
        OUTPUT_DIR / class_name
    )

    class_output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    model_path = (
        class_output_dir
        / f"{class_name}_autoencoder.keras"
    )

    best_model_path = (
        class_output_dir
        / f"{class_name}_best_autoencoder.keras"
    )

    history_path = (
        class_output_dir
        / f"{class_name}_training_history.csv"
    )

    error_path = (
        class_output_dir
        / f"{class_name}_reconstruction_errors.npz"
    )

    callbacks = [

        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),

        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(best_model_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )
    ]

    print("\nStarting training...")

    history = model.fit(
        X_train,
        X_train,
        validation_data=(
            X_val,
            X_val
        ),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=True,
        callbacks=callbacks,
        verbose=1
    )

    # Restore the best validation-loss weights.
    model = tf.keras.models.load_model(
        best_model_path
    )

    # Save final model.
    model.save(model_path)

    # Save training history.
    with open(
        history_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "epoch",
            "loss",
            "mae",
            "val_loss",
            "val_mae"
        ])

        for index in range(
            len(history.history["loss"])
        ):

            writer.writerow([
                index + 1,
                history.history["loss"][index],
                history.history["mae"][index],
                history.history["val_loss"][index],
                history.history["val_mae"][index]
            ])

    # Calculate reconstruction errors.
    train_errors = calculate_reconstruction_errors(
        model,
        X_train
    )

    val_errors = calculate_reconstruction_errors(
        model,
        X_val
    )

    # Save reconstruction error distributions.
    np.savez_compressed(
        error_path,

        train_errors=train_errors,
        validation_errors=val_errors,

        train_mean=np.array(
            np.mean(train_errors)
        ),

        train_std=np.array(
            np.std(train_errors)
        ),

        validation_mean=np.array(
            np.mean(val_errors)
        ),

        validation_std=np.array(
            np.std(val_errors)
        )
    )

    print("\nTraining completed.")

    print(
        f"Best validation loss: "
        f"{min(history.history['val_loss']):.6f}"
    )

    print(
        f"Mean training reconstruction error: "
        f"{np.mean(train_errors):.6f}"
    )

    print(
        f"Mean validation reconstruction error: "
        f"{np.mean(val_errors):.6f}"
    )

    print(f"Saved model: {model_path}")

    return {
        "class": class_name,
        "train_samples": len(X_train),
        "validation_samples": len(X_val),
        "epochs_trained": len(history.history["loss"]),
        "best_validation_loss": float(
            min(history.history["val_loss"])
        ),
        "train_reconstruction_error": float(
            np.mean(train_errors)
        ),
        "validation_reconstruction_error": float(
            np.mean(val_errors)
        )
    }


# ============================================================
# MAIN
# ============================================================

def main():

    set_seeds()

    print("=" * 60)
    print("BADMINTON FORM EVALUATION - LSTM AUTOENCODERS")
    print("=" * 60)

    print(f"\nSelected classifier run: {RUN_DIR.name}")

    if not RUN_DIR.exists():
        raise FileNotFoundError(
            f"Selected run directory not found:\n{RUN_DIR}"
        )

    mean, std = load_normalisation()

    training_files = load_training_manifest()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    results = []

    for class_name in CLASSES:

        X = load_class_sequences(
            class_name,
            training_files[class_name],
            mean,
            std
        )

        result = train_class_autoencoder(
            class_name,
            X
        )

        results.append(result)

        # Release model memory before next class.
        tf.keras.backend.clear_session()

    # Save summary.
    summary_path = (
        OUTPUT_DIR / "training_summary.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file,
            indent=4
        )

    print("\n" + "=" * 60)
    print("ALL AUTOENCODERS TRAINED SUCCESSFULLY")
    print("=" * 60)

    for result in results:

        print(
            f"\n{result['class'].capitalize()}: "
            f"validation loss = "
            f"{result['best_validation_loss']:.6f}"
        )

    print(
        f"\nOutput directory:\n{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()

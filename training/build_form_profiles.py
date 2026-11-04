
import csv
from pathlib import Path

import numpy as np

from training.train_lstm_full_feature import (
    PROJECT_ROOT,
    CLASSES,
    NUM_FRAMES,
    NUM_POSE_FEATURES,
    NUM_BIOMECHANICAL_FEATURES,
    NUM_FEATURES,
    augment_sequence,
)


# ============================================================
# CONFIGURATION
# ============================================================

EVALUATION_DIR = (
    PROJECT_ROOT
    / "evaluation"
    / "700_per_class_full_features"
)

# Selected run directory name containing the training split.
RUN_DIR_NAME = "run_059"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "models"
    / "form_evaluation"
    / "reference_profiles"
)

EXPECTED_TRAIN_SAMPLES_PER_CLASS = 490

# Feature index ranges
BIO_START = 99
BIO_END = 111

VELOCITY_START = 111
VELOCITY_END = 119

ACCELERATION_START = 119
ACCELERATION_END = 127

TIMING_START = 127
TIMING_END = 139

PERCENTILES = [5, 50, 95]


# ============================================================
# FIND THE CORRECT RUN DIRECTORY
# ============================================================

def get_run_directory():

    if RUN_DIR_NAME is not None:

        run_dir = EVALUATION_DIR / RUN_DIR_NAME

        if not run_dir.exists():
            raise FileNotFoundError(
                f"Run directory not found: {run_dir}"
            )

        return run_dir

    candidates = sorted(
        folder
        for folder in EVALUATION_DIR.glob("run_*")
        if (folder / "data_split_manifest.csv").exists()
    )

    if len(candidates) == 0:
        raise FileNotFoundError(
            "No run directories containing "
            "data_split_manifest.csv were found."
        )

    if len(candidates) > 1:

        available = "\n".join(
            folder.name for folder in candidates
        )

        raise RuntimeError(
            "Multiple run directories were found.\n"
            "Set RUN_DIR_NAME to your selected seed 42 run.\n\n"
            f"Available runs:\n{available}"
        )

    return candidates[0]


# ============================================================
# LOAD TRAINING FILES FROM MANIFEST
# ============================================================

def load_training_manifest(run_dir):

    manifest_path = (
        run_dir / "data_split_manifest.csv"
    )

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}"
        )

    training_files = {
        class_name: []
        for class_name in CLASSES
    }

    with open(
        manifest_path,
        "r",
        newline="",
        encoding="utf-8"
    ) as csv_file:

        reader = csv.DictReader(csv_file)

        required_columns = {
            "file",
            "split",
            "class"
        }

        if not required_columns.issubset(
            reader.fieldnames or []
        ):
            raise ValueError(
                "Manifest must contain: "
                "file, split, class"
            )

        for row in reader:

            # Only use the training split.
            if row["split"].strip() != "train":
                continue

            class_name = row["class"].strip()

            if class_name not in CLASSES:
                raise ValueError(
                    f"Unknown class in manifest: {class_name}"
                )

            relative_path = Path(row["file"])

            file_path = PROJECT_ROOT / relative_path

            training_files[class_name].append(
                file_path
            )

    for class_name in CLASSES:

        count = len(training_files[class_name])

        if count != EXPECTED_TRAIN_SAMPLES_PER_CLASS:
            raise ValueError(
                f"{class_name}: expected "
                f"{EXPECTED_TRAIN_SAMPLES_PER_CLASS} "
                f"training files, found {count}."
            )

    total = sum(
        len(files)
        for files in training_files.values()
    )

    if total != 1470:
        raise ValueError(
            f"Expected 1470 training sequences, found {total}."
        )

    return training_files


# ============================================================
# LOAD AND AUGMENT TRAINING SEQUENCES
# ============================================================

def load_class_sequences(class_name, file_paths):

    sequences = []

    print(
        f"\nLoading {class_name}: "
        f"{len(file_paths)} sequences"
    )

    for index, file_path in enumerate(
        file_paths,
        start=1
    ):

        if not file_path.exists():
            raise FileNotFoundError(
                f"Sequence not found: {file_path}"
            )

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
                f"Invalid pose shape in {file_path}: "
                f"{pose_sequence.shape}"
            )

        # Generate the same 139 features used by
        # the existing LSTM classifier.
        full_sequence = augment_sequence(
            pose_sequence
        )

        if full_sequence.shape != (
            NUM_FRAMES,
            NUM_FEATURES
        ):
            raise ValueError(
                f"Invalid augmented shape: "
                f"{full_sequence.shape}"
            )

        if not np.isfinite(full_sequence).all():
            raise ValueError(
                f"Non-finite features in {file_path}"
            )

        sequences.append(full_sequence)

        if index % 100 == 0:
            print(
                f"  Processed {index}/"
                f"{len(file_paths)}"
            )

    X = np.asarray(
        sequences,
        dtype=np.float32
    )

    expected_shape = (
        EXPECTED_TRAIN_SAMPLES_PER_CLASS,
        NUM_FRAMES,
        NUM_FEATURES
    )

    if X.shape != expected_shape:
        raise ValueError(
            f"Unexpected dataset shape for {class_name}: "
            f"{X.shape}, expected {expected_shape}"
        )

    return X


# ============================================================
# BUILD CLASS REFERENCE PROFILE
# ============================================================

def build_reference_profile(class_name, X):

    # All features across all videos and frames.
    all_features = X.reshape(
        -1,
        NUM_FEATURES
    )

    feature_mean = np.mean(
        all_features,
        axis=0
    )

    feature_std = np.std(
        all_features,
        axis=0
    )

    feature_percentiles = np.percentile(
        all_features,
        PERCENTILES,
        axis=0
    )

    # Biomechanical features: 99-110.
    bio_features = X[
        :,
        :,
        BIO_START:BIO_END
    ].reshape(
        -1,
        NUM_BIOMECHANICAL_FEATURES
    )

    bio_mean = np.mean(
        bio_features,
        axis=0
    )

    bio_std = np.std(
        bio_features,
        axis=0
    )

    bio_percentiles = np.percentile(
        bio_features,
        PERCENTILES,
        axis=0
    )

    # Angular velocity features.
    velocity = X[
        :,
        :,
        VELOCITY_START:VELOCITY_END
    ]

    velocity_mean = np.mean(
        velocity,
        axis=(0, 1)
    )

    velocity_std = np.std(
        velocity,
        axis=(0, 1)
    )

    # Angular acceleration features.
    acceleration = X[
        :,
        :,
        ACCELERATION_START:ACCELERATION_END
    ]

    acceleration_mean = np.mean(
        acceleration,
        axis=(0, 1)
    )

    acceleration_std = np.std(
        acceleration,
        axis=(0, 1)
    )

    # Timing features are repeated across all 30 frames.
    # Take frame 0 so every video contributes only once.
    timing = X[
        :,
        0,
        TIMING_START:TIMING_END
    ]

    timing_mean = np.mean(
        timing,
        axis=0
    )

    timing_std = np.std(
        timing,
        axis=0
    )

    timing_percentiles = np.percentile(
        timing,
        PERCENTILES,
        axis=0
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        OUTPUT_DIR
        / f"{class_name}_reference.npz"
    )

    np.savez_compressed(
        output_path,

        class_name=np.array(class_name),

        sample_count=np.array(len(X)),

        feature_mean=feature_mean,
        feature_std=feature_std,
        feature_percentiles=feature_percentiles,

        bio_mean=bio_mean,
        bio_std=bio_std,
        bio_percentiles=bio_percentiles,

        velocity_mean=velocity_mean,
        velocity_std=velocity_std,

        acceleration_mean=acceleration_mean,
        acceleration_std=acceleration_std,

        timing_mean=timing_mean,
        timing_std=timing_std,
        timing_percentiles=timing_percentiles,
    )

    print(
        f"Saved {class_name} reference profile:"
        f"\n{output_path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("BUILDING BADMINTON FORM REFERENCE PROFILES")
    print("=" * 60)

    run_dir = get_run_directory()

    print(f"\nSelected run: {run_dir.name}")

    training_files = load_training_manifest(
        run_dir
    )

    print("\nTraining split verified.")
    print("Clear: 490")
    print("Drop: 490")
    print("Smash: 490")
    print("Total: 1470")

    for class_name in CLASSES:

        X = load_class_sequences(
            class_name,
            training_files[class_name]
        )

        build_reference_profile(
            class_name,
            X
        )

    print("\n" + "=" * 60)
    print("REFERENCE PROFILE GENERATION COMPLETED")
    print("=" * 60)

    print(f"\nOutput directory:\n{OUTPUT_DIR}")


if __name__ == "__main__":
    main()

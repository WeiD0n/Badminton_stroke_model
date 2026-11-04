
import csv
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split

from training.train_form_autoencoder import (
    SEED,
    CLASSES,
    NUM_FRAMES,
    NUM_FEATURES,
    BATCH_SIZE,
    OUTPUT_DIR,
    load_normalisation,
    load_training_manifest,
)

from training.train_lstm_full_feature import augment_sequence


# ============================================================
# CONFIGURATION
# ============================================================

ANALYSIS_DIR = OUTPUT_DIR / "analysis"
TOP_N = 10


# ============================================================
# LOAD ONE SEQUENCE
# ============================================================

def load_one_sequence(file_path, mean, std):

    sequence = np.asarray(
        np.load(file_path, allow_pickle=False),
        dtype=np.float32
    )

    if sequence.shape != (NUM_FRAMES, 99):
        raise ValueError(
            f"Invalid pose sequence shape in {file_path}: "
            f"{sequence.shape}"
        )

    sequence = augment_sequence(sequence)

    if sequence.shape != (NUM_FRAMES, NUM_FEATURES):
        raise ValueError(
            f"Invalid augmented shape in {file_path}: "
            f"{sequence.shape}"
        )

    if not np.isfinite(sequence).all():
        raise ValueError(
            f"Non-finite values in {file_path}"
        )

    sequence = (sequence - mean) / std

    return sequence.astype(np.float32)


# ============================================================
# LOAD AVAILABLE SEQUENCES WITHOUT CHANGING SPLIT MEMBERSHIP
# ============================================================

def load_available_split(file_paths, indices, mean, std):

    sequences = []
    available_files = []
    missing_files = []

    for index in indices:

        file_path = file_paths[index]

        if not file_path.exists():
            missing_files.append(file_path)
            continue

        sequence = load_one_sequence(
            file_path,
            mean,
            std
        )

        sequences.append(sequence)
        available_files.append(file_path)

    if not sequences:
        raise ValueError(
            "No available sequences found in this split."
        )

    X = np.asarray(
        sequences,
        dtype=np.float32
    )

    return X, available_files, missing_files


# ============================================================
# RECONSTRUCTION ERRORS
# ============================================================

def calculate_errors(model, X):

    reconstructed = model.predict(
        X,
        batch_size=BATCH_SIZE,
        verbose=0
    )

    errors = np.mean(
        np.square(X - reconstructed),
        axis=(1, 2)
    )

    return errors.astype(np.float64)


# ============================================================
# ERROR STATISTICS
# ============================================================

def calculate_statistics(errors):

    q1 = float(np.percentile(errors, 25))
    median = float(np.percentile(errors, 50))
    q3 = float(np.percentile(errors, 75))

    iqr = q3 - q1
    upper_fence = q3 + (1.5 * iqr)

    return {
        "count": int(len(errors)),
        "mean": float(np.mean(errors)),
        "std": float(np.std(errors)),
        "min": float(np.min(errors)),
        "max": float(np.max(errors)),
        "percentile_25": q1,
        "median": median,
        "percentile_75": q3,
        "percentile_90": float(np.percentile(errors, 90)),
        "percentile_95": float(np.percentile(errors, 95)),
        "percentile_99": float(np.percentile(errors, 99)),
        "iqr": float(iqr),
        "upper_outlier_fence": float(upper_fence),
        "potential_outlier_count": int(
            np.sum(errors > upper_fence)
        )
    }


# ============================================================
# MAIN ANALYSIS
# ============================================================

def main():

    print("=" * 65)
    print("BADMINTON FORM AUTOENCODER ERROR ANALYSIS")
    print("=" * 65)

    ANALYSIS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    mean, std = load_normalisation()
    training_files = load_training_manifest()

    all_records = []
    summary = {}

    for class_name in CLASSES:

        print("\n" + "=" * 65)
        print(f"ANALYSING {class_name.upper()}")
        print("=" * 65)

        file_paths = training_files[class_name]

        # Reproduce the original 80/20 split using the
        # complete manifest, including quarantined files.
        indices = np.arange(len(file_paths))

        train_indices, val_indices = train_test_split(
            indices,
            test_size=0.20,
            random_state=SEED,
            shuffle=True
        )

        # Only now skip missing files. This does not change
        # the original train/validation membership.
        X_train, files_train, missing_train = (
            load_available_split(
                file_paths,
                train_indices,
                mean,
                std
            )
        )

        X_val, files_val, missing_val = (
            load_available_split(
                file_paths,
                val_indices,
                mean,
                std
            )
        )

        missing_files = missing_train + missing_val

        print(f"Original manifest count: {len(file_paths)}")
        print(f"Available training sequences: {len(X_train)}")
        print(f"Available validation sequences: {len(X_val)}")
        print(f"Missing/quarantined sequences: {len(missing_files)}")

        for missing_file in missing_files:
            print(f"  Skipped: {missing_file.name}")

        # Load the existing best model.
        model_path = (
            OUTPUT_DIR
            / class_name
            / f"{class_name}_best_autoencoder.keras"
        )

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {model_path}"
            )

        model = tf.keras.models.load_model(model_path)

        # Calculate reconstruction errors.
        train_errors = calculate_errors(model, X_train)
        val_errors = calculate_errors(model, X_val)

        # Calculate statistics.
        train_stats = calculate_statistics(train_errors)
        val_stats = calculate_statistics(val_errors)

        summary[class_name] = {
            "original_manifest_count": len(file_paths),
            "training": train_stats,
            "validation": val_stats,
            "missing_files": [
                file_path.name for file_path in missing_files
            ],
            "missing_training_count": len(missing_train),
            "missing_validation_count": len(missing_val),
            "validation_to_training_mean_ratio": float(
                val_stats["mean"] /
                max(train_stats["mean"], 1e-12)
            )
        }

        print("\nTRAINING ERROR")
        print(json.dumps(train_stats, indent=4))

        print("\nVALIDATION ERROR")
        print(json.dumps(val_stats, indent=4))

        # Save individual training errors.
        for file_path, error in zip(files_train, train_errors):

            all_records.append({
                "class": class_name,
                "split": "train",
                "filename": file_path.name,
                "file_path": str(file_path),
                "reconstruction_error": float(error)
            })

        # Save individual validation errors.
        for file_path, error in zip(files_val, val_errors):

            all_records.append({
                "class": class_name,
                "split": "validation",
                "filename": file_path.name,
                "file_path": str(file_path),
                "reconstruction_error": float(error)
            })

        # Display highest-error validation sequences.
        print(f"\nTOP {TOP_N} VALIDATION ERRORS")

        sorted_indices = np.argsort(val_errors)[::-1]

        for rank, index in enumerate(
            sorted_indices[:TOP_N],
            start=1
        ):

            print(
                f"{rank:2}. "
                f"{files_val[index].name:25} "
                f"Error: {val_errors[index]:.6f}"
            )

        tf.keras.backend.clear_session()

    # ========================================================
    # SAVE INDIVIDUAL ERRORS
    # ========================================================

    csv_path = ANALYSIS_DIR / "reconstruction_error_details.csv"

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        fieldnames = [
            "class",
            "split",
            "filename",
            "file_path",
            "reconstruction_error"
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(all_records)

    # ========================================================
    # SAVE SUMMARY
    # ========================================================

    summary_path = ANALYSIS_DIR / "reconstruction_error_summary.json"

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            summary,
            file,
            indent=4
        )

    # ========================================================
    # FINAL COMPARISON
    # ========================================================

    print("\n" + "=" * 75)
    print("FINAL RECONSTRUCTION ERROR COMPARISON")
    print("=" * 75)

    print(
        f"{'Class':<12}"
        f"{'Train N':>9}"
        f"{'Val N':>9}"
        f"{'Train Mean':>14}"
        f"{'Val Mean':>14}"
        f"{'Val/Train':>12}"
        f"{'Val Outliers':>15}"
    )

    for class_name in CLASSES:

        class_summary = summary[class_name]
        train_stats = class_summary["training"]
        val_stats = class_summary["validation"]

        print(
            f"{class_name.capitalize():<12}"
            f"{train_stats['count']:>9}"
            f"{val_stats['count']:>9}"
            f"{train_stats['mean']:>14.6f}"
            f"{val_stats['mean']:>14.6f}"
            f"{class_summary['validation_to_training_mean_ratio']:>12.2f}"
            f"{val_stats['potential_outlier_count']:>15}"
        )

    print("\nAnalysis completed.")
    print(f"\nDetailed errors:\n{csv_path}")
    print(f"\nSummary:\n{summary_path}")


if __name__ == "__main__":
    main()

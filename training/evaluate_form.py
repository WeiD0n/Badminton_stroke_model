import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from training.train_lstm_full_feature import (
    PROJECT_ROOT,
    CLASSES,
    NUM_FRAMES,
    NUM_FEATURES,
    augment_sequence,
)

# ============================================================
# CONFIGURATION
# ============================================================

REFERENCE_DIR = (
    PROJECT_ROOT
    / "models"
    / "form_evaluation"
    / "reference_profiles"
)

AUTOENCODER_DIR = (
    PROJECT_ROOT
    / "models"
    / "form_evaluation"
    / "autoencoders"
)

RUN_DIR = (
    PROJECT_ROOT
    / "evaluation"
    / "700_per_class_full_features"
    / "run_059"
)

NORMALISATION_PATH = RUN_DIR / "normalisation_stats.npz"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "evaluation"
    / "form_evaluation_results"
)

# Prototype weighting. These are design choices, not experimentally
# validated weights. Keep unchanged for compatibility with existing scores.
BIOMECHANICAL_WEIGHT = 0.50
TEMPORAL_WEIGHT = 0.50

# Feature indices in the 139-feature sequence.
BIO_START, BIO_END = 99, 111
VELOCITY_START, VELOCITY_END = 111, 119
ACCELERATION_START, ACCELERATION_END = 119, 127
TIMING_START, TIMING_END = 127, 139

# Names follow the exact order produced by train_lstm_full_feature.py.
FEATURE_NAMES = [
    "Left elbow angle",
    "Right elbow angle",
    "Left shoulder angle",
    "Right shoulder angle",
    "Left hip angle",
    "Right hip angle",
    "Left knee angle",
    "Right knee angle",
    "Torso lean (depth proxy)",
    "Torso lean (lateral proxy)",
    "Left elbow angle change",
    "Right elbow angle change",
    "Left elbow angular velocity",
    "Right elbow angular velocity",
    "Left shoulder angular velocity",
    "Right shoulder angular velocity",
    "Left hip angular velocity",
    "Right hip angular velocity",
    "Left knee angular velocity",
    "Right knee angular velocity",
    "Left elbow angular acceleration",
    "Right elbow angular acceleration",
    "Left shoulder angular acceleration",
    "Right shoulder angular acceleration",
    "Left hip angular acceleration",
    "Right hip angular acceleration",
    "Left knee angular acceleration",
    "Right knee angular acceleration",
    "Left elbow peak-angle timing",
    "Right elbow peak-angle timing",
    "Left shoulder peak-angle timing",
    "Right shoulder peak-angle timing",
    "Left hip peak-angle timing",
    "Right hip peak-angle timing",
    "Left knee peak-angle timing",
    "Right knee peak-angle timing",
    "Left elbow peak-angular-velocity timing",
    "Right elbow peak-angular-velocity timing",
    "Left shoulder peak-angular-velocity timing",
    "Right shoulder peak-angular-velocity timing",
]

JOINT_ANGLE_NAMES = FEATURE_NAMES[:8]
TORSO_NAMES = FEATURE_NAMES[8:10]
ELBOW_CHANGE_NAMES = FEATURE_NAMES[10:12]
VELOCITY_NAMES = FEATURE_NAMES[12:20]
ACCELERATION_NAMES = FEATURE_NAMES[20:28]
TIMING_NAMES = FEATURE_NAMES[28:40]

# Deviation thresholds are expressed in reference standard-deviation units.
# These are descriptive bands, not validated coaching thresholds.
CLOSE_DEVIATION = 1.0
MODERATE_DEVIATION = 2.0


# ============================================================
# LOAD NORMALISATION
# ============================================================

def load_normalisation():
    data = np.load(NORMALISATION_PATH, allow_pickle=False)

    mean = np.asarray(data["mean"], dtype=np.float32)
    std = np.asarray(data["std"], dtype=np.float32)

    if mean.shape != (NUM_FEATURES,):
        raise ValueError("Invalid normalization mean shape.")

    if std.shape != (NUM_FEATURES,):
        raise ValueError("Invalid normalization std shape.")

    std[std < 1e-7] = 1.0
    return mean, std


# ============================================================
# LOAD CLASS REFERENCE PROFILE
# ============================================================

def load_reference(stroke_class):
    path = REFERENCE_DIR / f"{stroke_class}_reference.npz"

    if not path.exists():
        raise FileNotFoundError(f"Reference profile not found: {path}")

    return np.load(path, allow_pickle=False)


# ============================================================
# LOAD CLASS AUTOENCODER
# ============================================================

def load_autoencoder(stroke_class):
    path = (
        AUTOENCODER_DIR
        / stroke_class
        / f"{stroke_class}_saved_model"
    )

    if not path.is_dir():
        raise FileNotFoundError(f"Autoencoder SavedModel not found: {path}")

    return tf.keras.models.load_model(path, compile=False)


# ============================================================
# REFERENCE STATISTICS
# ============================================================

def _reference_statistics(reference):
    """Return reference medians and scales in the same 40-feature order."""
    medians = np.concatenate([
        np.asarray(reference["bio_percentiles"], dtype=np.float32)[1],
        np.asarray(reference["velocity_mean"], dtype=np.float32),
        np.asarray(reference["acceleration_mean"], dtype=np.float32),
        np.asarray(reference["timing_percentiles"], dtype=np.float32)[1],
    ])

    scales = np.concatenate([
        np.asarray(reference["bio_std"], dtype=np.float32),
        np.asarray(reference["velocity_std"], dtype=np.float32),
        np.asarray(reference["acceleration_std"], dtype=np.float32),
        np.asarray(reference["timing_std"], dtype=np.float32),
    ])

    if medians.shape != (40,) or scales.shape != (40,):
        raise ValueError(
            "Reference profile must contain 40 biomechanical/temporal "
            "feature statistics. Check the reference-profile generation."
        )

    scales = np.maximum(scales, 1e-6)
    return medians, scales


def _sequence_feature_values(sequence):
    """Aggregate frame features while counting repeated timing values once."""
    return np.concatenate([
        np.mean(sequence[:, BIO_START:BIO_END], axis=0),
        np.mean(sequence[:, VELOCITY_START:VELOCITY_END], axis=0),
        np.mean(sequence[:, ACCELERATION_START:ACCELERATION_END], axis=0),
        sequence[0, TIMING_START:TIMING_END],
    ]).astype(np.float32)


# ============================================================
# BIOMECHANICAL AND TEMPORAL COMPARISON
# ============================================================

def compare_with_reference(sequence, reference):
    values = _sequence_feature_values(sequence)
    reference_medians, reference_scales = _reference_statistics(reference)

    deviations = np.abs(values - reference_medians) / reference_scales
    deviations = np.clip(deviations, 0, 5)

    # Existing score formula retained to avoid changing dashboard metrics.
    mean_deviation = float(np.mean(deviations))
    score = 100.0 * np.exp(-mean_deviation)

    feature_results = []
    for index, deviation in enumerate(deviations):
        feature_results.append({
            "feature": FEATURE_NAMES[index],
            "deviation": float(deviation),
        })

    feature_results.sort(
        key=lambda item: item["deviation"],
        reverse=True,
    )

    return float(np.clip(score, 0, 100)), feature_results


# ============================================================
# DETAILED BIOMECHANICAL FEEDBACK
# ============================================================

def _deviation_status(deviation):
    if deviation < CLOSE_DEVIATION:
        return "close_to_reference"
    if deviation < MODERATE_DEVIATION:
        return "moderate_deviation"
    return "high_deviation"


def _measurement_detail(name, value, reference_value, scale, unit, category):
    deviation = abs(float(value) - float(reference_value)) / max(
        float(scale), 1e-6
    )
    status = _deviation_status(deviation)

    return {
        "feature": name,
        "category": category,
        "value": round(float(value), 3),
        "reference_value": round(float(reference_value), 3),
        "unit": unit,
        "deviation": round(float(deviation), 3),
        "status": status,
        "status_label": {
            "close_to_reference": "Close to reference",
            "moderate_deviation": "Moderate deviation",
            "high_deviation": "Larger deviation",
        }[status],
        "explanation": (
            f"Your measured {name.lower()} was {float(value):.1f} {unit}; "
            f"the reference median was {float(reference_value):.1f} {unit}. "
            f"The difference is {deviation:.2f} reference standard deviations."
        ),
    }


def generate_detailed_feedback(sequence, reference):
    """
    Create interpretable per-feature comparisons from the same reference
    statistics used by the existing form evaluator.

    Values are descriptive comparisons, not clinical or validated coaching
    judgements. Coordinates are hip-relative and camera-view dependent.
    """
    values = _sequence_feature_values(sequence)
    medians, scales = _reference_statistics(reference)

    details = []
    for index, name in enumerate(FEATURE_NAMES):
        if index < 10:
            category = "joint_angles" if index < 8 else "torso_alignment"
            unit = "degrees"
        elif index < 12:
            category = "elbow_angle_change"
            unit = "degrees_per_frame"
        elif index < 20:
            category = "angular_velocity"
            unit = "degrees_per_normalised_time"
        elif index < 28:
            category = "angular_acceleration"
            unit = "degrees_per_normalised_time_squared"
        else:
            category = "movement_timing"
            unit = "normalised_sequence_time"

        details.append(
            _measurement_detail(
                name=name,
                value=values[index],
                reference_value=medians[index],
                scale=scales[index],
                unit=unit,
                category=category,
            )
        )

    joint_details = details[:8]
    torso_details = details[8:10]
    elbow_change_details = details[10:12]
    velocity_details = details[12:20]
    acceleration_details = details[20:28]
    timing_details = details[28:40]

    # "Strengths" here means measurements relatively close to the selected
    # stroke reference, not proof that the technique is objectively correct.
    strengths = [
        {
            "feature": item["feature"],
            "message": (
                f"{item['feature']} is relatively close to the "
                "stroke-specific reference profile."
            ),
        }
        for item in details
        if item["status"] == "close_to_reference"
    ]

    areas_to_review = [
        {
            "feature": item["feature"],
            "message": (
                f"Review {item['feature'].lower()}: its measured value "
                "differs more from the stroke-specific reference profile."
            ),
            "deviation": item["deviation"],
            "status": item["status"],
        }
        for item in sorted(
            details,
            key=lambda item: item["deviation"],
            reverse=True,
        )
        if item["status"] != "close_to_reference"
    ][:5]

    return {
        "summary": (
            "Measurements are compared with the reference profile for this "
            "stroke. The deviation describes statistical distance from the "
            "reference, not whether a movement is inherently correct or incorrect."
        ),
        "joint_angles": joint_details,
        "torso_alignment": torso_details,
        "elbow_angle_changes": elbow_change_details,
        "angular_velocity": velocity_details,
        "angular_acceleration": acceleration_details,
        "movement_timing": timing_details,
        "strengths": strengths,
        "areas_to_review": areas_to_review,
        "all_measurements": details,
        "interpretation": (
            "Reference-profile comparison is sensitive to camera angle, "
            "visibility, player orientation, tracking quality and the "
            "reference dataset. Treat these results as movement-analysis "
            "indicators rather than validated coaching or medical advice."
        ),
    }


# ============================================================
# LEGACY FEEDBACK (PRESERVED FOR EXISTING FRONTEND)
# ============================================================

def generate_feedback(feature_results):
    feedback = []

    for item in feature_results[:3]:
        if item["deviation"] < CLOSE_DEVIATION:
            continue

        feature = item["feature"]
        feedback.append(
            f"Review your {feature.lower()} compared "
            "with the professional reference movement."
        )

    if not feedback:
        feedback.append(
            "Your measured features are relatively close "
            "to the professional reference profile."
        )

    return feedback


# ============================================================
# AUTOENCODER RECONSTRUCTION SCORE
# ============================================================

def calculate_temporal_score(sequence, stroke_class, mean, std):
    model = load_autoencoder(stroke_class)

    normalized = (sequence - mean) / std
    X = np.expand_dims(normalized.astype(np.float32), axis=0)

    reconstructed = model.predict(X, verbose=0)

    reconstruction_error = float(
        np.mean(np.square(X - reconstructed))
    )

    error_path = (
        AUTOENCODER_DIR
        / stroke_class
        / f"{stroke_class}_reconstruction_errors.npz"
    )

    if not error_path.exists():
        raise FileNotFoundError(
            f"Reconstruction error statistics not found: {error_path}"
        )

    error_data = np.load(error_path, allow_pickle=False)

    reference_mean = float(error_data["validation_mean"])
    reference_std = max(float(error_data["validation_std"]), 1e-8)

    z_score = (
        reconstruction_error - reference_mean
    ) / reference_std

    temporal_score = 100.0 * np.exp(
        -max(0.0, z_score) / 3.0
    )

    return (
        float(np.clip(temporal_score, 0, 100)),
        reconstruction_error,
    )


# ============================================================
# MAIN FORM EVALUATION FUNCTION
# ============================================================

def evaluate_form(pose_sequence, predicted_class):
    if predicted_class not in CLASSES:
        raise ValueError(f"Unknown stroke class: {predicted_class}")

    pose_sequence = np.asarray(pose_sequence, dtype=np.float32)

    if pose_sequence.shape != (NUM_FRAMES, 99):
        raise ValueError(
            f"Expected pose shape {(NUM_FRAMES, 99)}, "
            f"got {pose_sequence.shape}"
        )

    if not np.isfinite(pose_sequence).all():
        raise ValueError("Pose sequence contains non-finite values.")

    # Reuse existing Objective 2 feature extraction.
    full_sequence = augment_sequence(pose_sequence)

    # Load matching stroke-specific professional reference.
    reference = load_reference(predicted_class)

    biomechanical_score, feature_results = compare_with_reference(
        full_sequence,
        reference,
    )

    detailed_feedback = generate_detailed_feedback(
        full_sequence,
        reference,
    )

    # Load normalization and matching autoencoder.
    mean, std = load_normalisation()

    temporal_score, reconstruction_error = calculate_temporal_score(
        full_sequence,
        predicted_class,
        mean,
        std,
    )

    # Existing score combination retained.
    final_score = (
        BIOMECHANICAL_WEIGHT * biomechanical_score
        + TEMPORAL_WEIGHT * temporal_score
    )

    if final_score >= 75:
        rating = "Good"
    elif final_score >= 50:
        rating = "Moderate"
    else:
        rating = "Needs Improvement"

    feedback = generate_feedback(feature_results)

    # Preserve all existing keys and add detailed_feedback for the UI.
    result = {
        "stroke_type": predicted_class.capitalize(),
        "form_score": round(final_score, 2),
        "form_rating": rating,
        "biomechanical_score": round(biomechanical_score, 2),
        "temporal_similarity_score": round(temporal_score, 2),
        "reconstruction_error": round(reconstruction_error, 6),
        "feedback": feedback,
        "top_feature_deviations": feature_results[:5],
        "detailed_feedback": detailed_feedback,
        "score_interpretation": (
            "Prototype similarity score based on professional reference "
            "statistics and autoencoder reconstruction. Not a validated "
            "coaching or biomechanical quality score."
        ),
    }

    return result


# ============================================================
# EXAMPLE TEST
# ============================================================

if __name__ == "__main__":
    sample_path = (
        PROJECT_ROOT
        / "dataset"
        / "sequences"
        / "smash"
        / "smash_001.npy"
    )

    sample = np.load(sample_path, allow_pickle=False)

    result = evaluate_form(
        sample,
        predicted_class="smash",
    )

    print(json.dumps(result, indent=4))

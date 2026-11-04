
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

# Prototype weighting. These are design choices,
# not experimentally validated weights.
BIOMECHANICAL_WEIGHT = 0.50
TEMPORAL_WEIGHT = 0.50

# Feature indices in the 139-feature sequence.
BIO_START, BIO_END = 99, 111
VELOCITY_START, VELOCITY_END = 111, 119
ACCELERATION_START, ACCELERATION_END = 119, 127
TIMING_START, TIMING_END = 127, 139

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
    "Torso depth peak-angle timing",
    "Torso lateral peak-angle timing",
    "Left elbow peak-velocity timing",
    "Right elbow peak-velocity timing",
    "Left shoulder peak-velocity timing",
    "Right shoulder peak-velocity timing",
]


# ============================================================
# LOAD NORMALISATION
# ============================================================

def load_normalisation():

    data = np.load(
        NORMALISATION_PATH,
        allow_pickle=False
    )

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
        raise FileNotFoundError(
            f"Reference profile not found: {path}"
        )

    return np.load(path, allow_pickle=False)


# ============================================================
# LOAD CLASS AUTOENCODER
# ============================================================

def load_autoencoder(stroke_class):

    path = (
        AUTOENCODER_DIR
        / stroke_class
        / f"{stroke_class}_best_autoencoder.keras"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Autoencoder not found: {path}"
        )

    return tf.keras.models.load_model(path, compile=False)


# ============================================================
# BIOMECHANICAL AND TEMPORAL COMPARISON
# ============================================================

def compare_with_reference(sequence, reference):

    # Aggregate frame-level features per sequence.
    bio_values = np.mean(
        sequence[:, BIO_START:BIO_END],
        axis=0
    )

    velocity_values = np.mean(
        sequence[:, VELOCITY_START:VELOCITY_END],
        axis=0
    )

    acceleration_values = np.mean(
        sequence[:, ACCELERATION_START:ACCELERATION_END],
        axis=0
    )

    # Timing features are repeated across frames.
    # Count them once per sequence.
    timing_values = sequence[0, TIMING_START:TIMING_END]

    values = np.concatenate([
        bio_values,
        velocity_values,
        acceleration_values,
        timing_values
    ])

    reference_medians = np.concatenate([
        reference["bio_percentiles"][1],
        reference["velocity_mean"],
        reference["acceleration_mean"],
        reference["timing_percentiles"][1]
    ])

    # Use reference standard deviations as scale estimates.
    reference_scales = np.concatenate([
        reference["bio_std"],
        reference["velocity_std"],
        reference["acceleration_std"],
        reference["timing_std"]
    ])

    # Avoid division by zero.
    reference_scales = np.maximum(
        reference_scales,
        1e-6
    )

    deviations = np.abs(
        values - reference_medians
    ) / reference_scales

    # Limit extreme contributions.
    deviations = np.clip(deviations, 0, 5)

    # Mean deviation across all 40 features.
    mean_deviation = float(np.mean(deviations))

    # Convert deviation into a prototype similarity score.
    score = 100.0 * np.exp(-mean_deviation)

    feature_results = []

    for index, deviation in enumerate(deviations):

        feature_results.append({
            "feature": FEATURE_NAMES[index],
            "deviation": float(deviation)
        })

    feature_results.sort(
        key=lambda item: item["deviation"],
        reverse=True
    )

    return float(np.clip(score, 0, 100)), feature_results


# ============================================================
# AUTOENCODER RECONSTRUCTION SCORE
# ============================================================

def calculate_temporal_score(sequence, stroke_class, mean, std):

    model = load_autoencoder(stroke_class)

    normalized = (sequence - mean) / std

    X = np.expand_dims(
        normalized.astype(np.float32),
        axis=0
    )

    reconstructed = model.predict(
        X,
        verbose=0
    )

    reconstruction_error = float(
        np.mean(np.square(X - reconstructed))
    )

    # Compare against the class-specific validation
    # error distribution saved during training.
    error_path = (
        AUTOENCODER_DIR
        / stroke_class
        / f"{stroke_class}_reconstruction_errors.npz"
    )

    if not error_path.exists():
        raise FileNotFoundError(
            f"Reconstruction error statistics not found: "
            f"{error_path}"
        )

    error_data = np.load(
        error_path,
        allow_pickle=False
    )

    reference_mean = float(error_data["validation_mean"])
    reference_std = max(
        float(error_data["validation_std"]),
        1e-8
    )

    # This is a relative anomaly score, not a validated
    # measure of actual badminton technique quality.
    z_score = (
        reconstruction_error - reference_mean
    ) / reference_std

    temporal_score = 100.0 * np.exp(
        -max(0.0, z_score) / 3.0
    )

    return (
        float(np.clip(temporal_score, 0, 100)),
        reconstruction_error
    )


# ============================================================
# GENERATE FEEDBACK
# ============================================================

def generate_feedback(feature_results):

    feedback = []

    for item in feature_results[:3]:

        if item["deviation"] < 1.0:
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
# MAIN FORM EVALUATION FUNCTION
# ============================================================

def evaluate_form(pose_sequence, predicted_class):

    if predicted_class not in CLASSES:
        raise ValueError(
            f"Unknown stroke class: {predicted_class}"
        )

    pose_sequence = np.asarray(
        pose_sequence,
        dtype=np.float32
    )

    if pose_sequence.shape != (NUM_FRAMES, 99):
        raise ValueError(
            f"Expected pose shape {(NUM_FRAMES, 99)}, "
            f"got {pose_sequence.shape}"
        )

    if not np.isfinite(pose_sequence).all():
        raise ValueError(
            "Pose sequence contains non-finite values."
        )

    # Reuse existing Objective 2 feature extraction.
    full_sequence = augment_sequence(pose_sequence)

    # Load matching professional reference.
    reference = load_reference(predicted_class)

    biomechanical_score, feature_results = (
        compare_with_reference(
            full_sequence,
            reference
        )
    )

    # Load normalization and matching autoencoder.
    mean, std = load_normalisation()

    temporal_score, reconstruction_error = (
        calculate_temporal_score(
            full_sequence,
            predicted_class,
            mean,
            std
        )
    )

    # Combine both components.
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

    result = {
        "stroke_type": predicted_class.capitalize(),
        "form_score": round(final_score, 2),
        "form_rating": rating,
        "biomechanical_score": round(
            biomechanical_score, 2
        ),
        "temporal_similarity_score": round(
            temporal_score, 2
        ),
        "reconstruction_error": round(
            reconstruction_error, 6
        ),
        "feedback": feedback,
        "top_feature_deviations": feature_results[:5],
        "score_interpretation": (
            "Prototype similarity score based on professional "
            "reference statistics and autoencoder reconstruction. "
            "Not a validated coaching or biomechanical quality score."
        )
    }

    return result


# ============================================================
# EXAMPLE TEST
# ============================================================

if __name__ == "__main__":

    # Replace with a real extracted pose sequence.
    sample_path = (
        PROJECT_ROOT
        / "dataset"
        / "sequences"
        / "smash"
        / "smash_001.npy"
    )

    sample = np.load(
        sample_path,
        allow_pickle=False
    )

    result = evaluate_form(
        sample,
        predicted_class="smash"
    )

    print(json.dumps(result, indent=4))

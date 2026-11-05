"""Run-059 LSTM badminton stroke classifier.

Preprocessing mirrors train_lstm_stroke_classifier_700_full_features_versioned.py:
30 x 99 pose -> 12 biomechanical + 28 temporal -> 30 x 139,
then training-set feature-wise z-score normalisation.
"""
from functools import lru_cache
from pathlib import Path

import numpy as np
import tensorflow as tf

NUM_FRAMES = 30
NUM_POSE_FEATURES = 99
NUM_BIOMECHANICAL_FEATURES = 12
NUM_TEMPORAL_FEATURES = 28
NUM_FEATURES = 139
EPSILON = 1e-7
CLASSES = ("clear", "drop", "smash")
DISPLAY_NAMES = {
    "clear": "Clear",
    "drop": "Drop Shot",
    "smash": "Smash",
}

# MediaPipe Pose landmark indices.
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_ELBOW, RIGHT_ELBOW = 13, 14
LEFT_WRIST, RIGHT_WRIST = 15, 16
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_KNEE, RIGHT_KNEE = 25, 26
LEFT_ANKLE, RIGHT_ANKLE = 27, 28


def calculate_angle(a, b, c):
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
    sequence = np.asarray(sequence, dtype=np.float32)
    if sequence.shape != (NUM_FRAMES, NUM_POSE_FEATURES):
        raise ValueError(f"Expected pose sequence {(NUM_FRAMES, NUM_POSE_FEATURES)}, got {sequence.shape}")

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
        bio[i, 8] = np.degrees(np.arctan2(torso[2], max(abs(torso[1]), EPSILON)))
        bio[i, 9] = np.degrees(np.arctan2(torso[0], max(abs(torso[1]), EPSILON)))

    bio[0, 10:12] = 0.0
    bio[1:, 10] = np.diff(bio[:, 0])
    bio[1:, 11] = np.diff(bio[:, 1])
    return np.nan_to_num(bio, nan=0.0, posinf=0.0, neginf=0.0)


def calculate_temporal_features(biomechanical):
    angles = np.asarray(biomechanical[:, :8], dtype=np.float32)
    time_axis = np.linspace(0.0, 1.0, NUM_FRAMES, dtype=np.float32)
    velocity = np.gradient(angles, time_axis, axis=0).astype(np.float32)
    acceleration = np.gradient(velocity, time_axis, axis=0).astype(np.float32)

    angle_peak_timing = np.zeros(8, dtype=np.float32)
    for j in range(8):
        angle_peak_timing[j] = time_axis[int(np.argmax(angles[:, j]))]

    velocity_joint_indices = [0, 1, 2, 3]
    velocity_peak_timing = np.zeros(4, dtype=np.float32)
    for out_idx, joint_idx in enumerate(velocity_joint_indices):
        velocity_peak_timing[out_idx] = time_axis[int(np.argmax(np.abs(velocity[:, joint_idx])))]

    timing = np.concatenate([angle_peak_timing, velocity_peak_timing]).astype(np.float32)
    timing_per_frame = np.repeat(timing[None, :], NUM_FRAMES, axis=0)
    temporal = np.concatenate([velocity, acceleration, timing_per_frame], axis=1)
    if temporal.shape != (NUM_FRAMES, NUM_TEMPORAL_FEATURES):
        raise ValueError(f"Unexpected temporal feature shape: {temporal.shape}")
    return np.nan_to_num(temporal, nan=0.0, posinf=0.0, neginf=0.0)


def build_classifier_features(pose_sequence):
    pose_sequence = np.asarray(pose_sequence, dtype=np.float32)
    if pose_sequence.shape != (NUM_FRAMES, NUM_POSE_FEATURES):
        raise ValueError(f"Expected pose input {(NUM_FRAMES, NUM_POSE_FEATURES)}, got {pose_sequence.shape}")
    if not np.isfinite(pose_sequence).all():
        raise ValueError("Pose sequence contains NaN or infinite values.")

    biomechanical = calculate_biomechanical_features(pose_sequence)
    temporal = calculate_temporal_features(biomechanical)
    augmented = np.concatenate([pose_sequence, biomechanical, temporal], axis=1)
    if augmented.shape != (NUM_FRAMES, NUM_FEATURES) or not np.isfinite(augmented).all():
        raise ValueError(f"Invalid full-feature sequence: {augmented.shape}")
    return augmented


class Run059Classifier:
    def __init__(self, project_root: Path):
        run_dir = Path(project_root) / "evaluation" / "700_per_class_full_features" / "run_059"
        model_path = run_dir / "final_model.keras"
        stats_path = run_dir / "normalisation_stats.npz"
        if not model_path.is_file():
            raise FileNotFoundError(f"Run-059 classifier model not found: {model_path}")
        if not stats_path.is_file():
            raise FileNotFoundError(f"Run-059 normalisation file not found: {stats_path}")

        self.model = tf.keras.models.load_model(model_path, compile=False)
        with np.load(stats_path, allow_pickle=False) as stats:
            self.mean = np.asarray(stats["mean"], dtype=np.float32).reshape(1, 1, -1)
            self.std = np.asarray(stats["std"], dtype=np.float32).reshape(1, 1, -1)
            saved_classes = tuple(str(x) for x in stats["classes"].tolist())
            saved_frames = int(np.asarray(stats["num_frames"]).item())
            saved_pose_features = int(np.asarray(stats["num_pose_features"]).item())
            saved_bio_features = int(np.asarray(stats["num_biomechanical_features"]).item())
            saved_temporal_features = int(np.asarray(stats["num_temporal_features"]).item())

        if saved_classes != CLASSES:
            raise ValueError(f"Run-059 class order mismatch: expected {CLASSES}, got {saved_classes}")
        if (saved_frames, saved_pose_features, saved_bio_features, saved_temporal_features) != (30, 99, 12, 28):
            raise ValueError("Run-059 normalisation metadata does not match the expected 30x139 feature pipeline.")
        if self.mean.shape != (1, 1, NUM_FEATURES) or self.std.shape != (1, 1, NUM_FEATURES):
            raise ValueError("Run-059 mean/std arrays must each contain 139 values.")
        self.std[self.std < 1e-7] = 1.0
        if tuple(self.model.input_shape[1:]) != (NUM_FRAMES, NUM_FEATURES):
            raise ValueError(f"Classifier input shape mismatch: {self.model.input_shape}")

    def predict(self, pose_sequence):
        features = build_classifier_features(pose_sequence)[None, ...]
        normalised = ((features - self.mean) / self.std).astype(np.float32)
        probabilities = np.asarray(self.model.predict(normalised, verbose=0)[0], dtype=np.float32)
        if probabilities.shape != (len(CLASSES),):
            raise ValueError(f"Unexpected classifier output shape: {probabilities.shape}")
        predicted_index = int(np.argmax(probabilities))
        predicted_class = CLASSES[predicted_index]
        return {
            "model": "LSTM Classifier",
            "run_id": "run_059",
            "predicted_class": predicted_class,
            "predicted_stroke": DISPLAY_NAMES[predicted_class],
            "confidence": float(probabilities[predicted_index]),
            "probabilities": {name: float(probabilities[i]) for i, name in enumerate(CLASSES)},
            "test_accuracy": 0.7015873015873015,
            "test_samples": 315,
        }


@lru_cache(maxsize=1)
def get_run059_classifier(project_root_string: str):
    return Run059Classifier(Path(project_root_string))


def predict_stroke(pose_sequence, project_root: Path):
    return get_run059_classifier(str(Path(project_root).resolve())).predict(pose_sequence)

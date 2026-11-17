
from pathlib import Path
import numpy as np

DATA_DIR = Path("dataset/sequences_tracked")
FRAMES = 30
BASE_FEATURES = 99

CLASSES = ["smash", "drop", "clear"]


def calculate_joint_angle(a, b, c):
    ba = a - b
    bc = c - b

    denominator = (
        np.linalg.norm(ba, axis=-1)
        * np.linalg.norm(bc, axis=-1)
    )

    dot = np.sum(ba * bc, axis=-1)
    cosine = np.zeros_like(dot, dtype=np.float32)

    valid = denominator > 1e-7
    cosine[valid] = dot[valid] / denominator[valid]
    cosine = np.clip(cosine, -1.0, 1.0)

    angles = np.degrees(np.arccos(cosine)).astype(np.float32)
    angles[~valid] = 0.0

    return angles, ~valid


def add_joint_angles(X):
    points = X.reshape(len(X), FRAMES, 33, 3)

    definitions = [
        ("Left elbow", 11, 13, 15),
        ("Right elbow", 12, 14, 16),
        ("Left knee", 23, 25, 27),
        ("Right knee", 24, 26, 28),
    ]

    angles = []
    invalid_masks = []

    for _, a, b, c in definitions:
        angle, invalid = calculate_joint_angle(
            points[:, :, a],
            points[:, :, b],
            points[:, :, c],
        )
        angles.append(angle)
        invalid_masks.append(invalid)

    angle_features = np.stack(angles, axis=-1)
    invalid_masks = np.stack(invalid_masks, axis=-1)

    X_combined = np.concatenate(
        [X, angle_features], axis=-1
    ).astype(np.float32)

    return X_combined, invalid_masks


# Load sequences
sequences = []

for class_name in CLASSES:
    folder = DATA_DIR / class_name

    for path in sorted(folder.glob("*.npy")):
        sequence = np.load(path, allow_pickle=False)

        if sequence.shape != (FRAMES, BASE_FEATURES):
            raise ValueError(
                f"{path}: unexpected shape {sequence.shape}"
            )

        if not np.isfinite(sequence).all():
            raise ValueError(f"NaN/Inf found in {path}")

        sequences.append(sequence.astype(np.float32))

X = np.asarray(sequences, dtype=np.float32)

if len(X) == 0:
    raise RuntimeError(f"No sequences found in {DATA_DIR}")

print("Sequences:", len(X))
print("Raw pose shape:", X.shape)

X_all, invalid_masks = add_joint_angles(X)

print("Pose + angle shape:", X_all.shape)

feature_names = [
    f"Pose feature {i}" for i in range(BASE_FEATURES)
] + [
    "Left elbow",
    "Right elbow",
    "Left knee",
    "Right knee",
]

print("\n--- RAW FEATURE RANGES ---")

for i, name in enumerate(feature_names):
    values = X_all[:, :, i]

    print(
        f"{name:18s} | "
        f"min={values.min():9.3f} "
        f"max={values.max():9.3f} "
        f"mean={values.mean():9.3f} "
        f"std={values.std():9.3f}"
    )

print("\n--- ANGLE VALIDITY ---")

for i, name in enumerate(feature_names[-4:]):
    invalid_count = int(invalid_masks[:, :, i].sum())
    total_count = invalid_masks[:, :, i].size

    print(
        f"{name:18s} | "
        f"invalid={invalid_count}/{total_count} "
        f"({100 * invalid_count / total_count:.2f}%)"
    )

# Diagnostic only: this uses all sequences to inspect ranges.
# Do not use these statistics to train or evaluate the model.
mean = X_all.mean(axis=(0, 1), keepdims=True)
std = X_all.std(axis=(0, 1), keepdims=True)
safe_std = np.where(std < 1e-7, 1.0, std)

X_scaled = (X_all - mean) / safe_std

print("\n--- AFTER Z-SCORE STANDARDIZATION ---")

for i, name in enumerate(feature_names):
    values = X_scaled[:, :, i]

    print(
        f"{name:18s} | "
        f"mean={values.mean():8.3f} "
        f"std={values.std():8.3f} "
        f"min={values.min():8.3f} "
        f"max={values.max():8.3f}"
    )

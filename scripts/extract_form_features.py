
from pathlib import Path
import csv
import numpy as np

DATA_DIR = Path("dataset/sequences_tracked")
OUTPUT_DIR = Path("dataset/form_features")

FRAMES = 30
LANDMARKS = 33
BASE_FEATURES = 99
CLASSES = ["smash", "drop", "clear"]

# MediaPipe landmark indices
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_ELBOW, RIGHT_ELBOW = 13, 14
LEFT_WRIST, RIGHT_WRIST = 15, 16
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_KNEE, RIGHT_KNEE = 25, 26
LEFT_ANKLE, RIGHT_ANKLE = 27, 28


def joint_angle(a, b, c):
    """Calculate the 3D joint angle at point b in degrees."""
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

    result = np.degrees(np.arccos(cosine)).astype(np.float32)
    result[~valid] = np.nan

    return result


def line_angle(a, b):
    """
    Calculate an undirected 2D body-line orientation.

    Output range: [-90, 90).
    Angles differing by 180 degrees represent the same line.
    """
    delta = b - a

    angle = np.degrees(
        np.arctan2(delta[..., 1], delta[..., 0])
    )

    angle = (angle + 90.0) % 180.0 - 90.0

    return angle.astype(np.float32)


def angle_difference(current, previous):
    """Return the shortest signed difference between two angles."""
    return (current - previous + 180.0) % 360.0 - 180.0


def extract_features(X):
    """
    Input:  (30, 99)
    Output: (30, 12)

    Feature order:
    0  left elbow angle
    1  right elbow angle
    2  left knee angle
    3  right knee angle
    4  trunk inclination from vertical
    5  shoulder-line orientation
    6  hip-line orientation
    7  left elbow angle change
    8  right elbow angle change
    9  left knee angle change
    10 right knee angle change
    11 trunk inclination change
    """
    points = X.reshape(FRAMES, LANDMARKS, 3)

    # Joint angles
    left_elbow = joint_angle(
        points[:, LEFT_SHOULDER],
        points[:, LEFT_ELBOW],
        points[:, LEFT_WRIST],
    )

    right_elbow = joint_angle(
        points[:, RIGHT_SHOULDER],
        points[:, RIGHT_ELBOW],
        points[:, RIGHT_WRIST],
    )

    left_knee = joint_angle(
        points[:, LEFT_HIP],
        points[:, LEFT_KNEE],
        points[:, LEFT_ANKLE],
    )

    right_knee = joint_angle(
        points[:, RIGHT_HIP],
        points[:, RIGHT_KNEE],
        points[:, RIGHT_ANKLE],
    )

    # Shoulder and hip midpoints
    shoulder_mid = (
        points[:, LEFT_SHOULDER]
        + points[:, RIGHT_SHOULDER]
    ) / 2.0

    hip_mid = (
        points[:, LEFT_HIP]
        + points[:, RIGHT_HIP]
    ) / 2.0

    # Trunk inclination in the image x-y plane.
    # 0 degrees means approximately vertical.
    trunk_vector = shoulder_mid[:, :2] - hip_mid[:, :2]

    trunk_inclination = np.degrees(
        np.arctan2(
            np.abs(trunk_vector[:, 0]),
            np.abs(trunk_vector[:, 1]) + 1e-7,
        )
    ).astype(np.float32)

    # Normalize line orientations to avoid the +/-180-degree jump.
    shoulder_orientation = line_angle(
        points[:, LEFT_SHOULDER, :2],
        points[:, RIGHT_SHOULDER, :2],
    )

    hip_orientation = line_angle(
        points[:, LEFT_HIP, :2],
        points[:, RIGHT_HIP, :2],
    )

    # Frame-to-frame joint-angle changes.
    # These are per sampled frame, not degrees per second.
    elbow_left_change = np.diff(
        left_elbow, prepend=left_elbow[0]
    )

    elbow_right_change = np.diff(
        right_elbow, prepend=right_elbow[0]
    )

    knee_left_change = np.diff(
        left_knee, prepend=left_knee[0]
    )

    knee_right_change = np.diff(
        right_knee, prepend=right_knee[0]
    )

    trunk_change = np.diff(
        trunk_inclination, prepend=trunk_inclination[0]
    )

    features = np.column_stack([
        left_elbow,
        right_elbow,
        left_knee,
        right_knee,
        trunk_inclination,
        shoulder_orientation,
        hip_orientation,
        elbow_left_change,
        elbow_right_change,
        knee_left_change,
        knee_right_change,
        trunk_change,
    ]).astype(np.float32)

    return features


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    records = []
    skipped = []

    for class_name in CLASSES:
        class_dir = DATA_DIR / class_name

        if not class_dir.exists():
            print(f"WARNING: Missing directory: {class_dir}")
            continue

        output_path = OUTPUT_DIR / class_name
        output_path.mkdir(parents=True, exist_ok=True)

        for path in sorted(class_dir.glob("*.npy")):
            try:
                X = np.load(path, allow_pickle=False)
            except (OSError, ValueError) as exc:
                skipped.append(
                    (class_name, path.name, f"Load error: {exc}")
                )
                continue

            if X.shape != (FRAMES, BASE_FEATURES):
                skipped.append(
                    (class_name, path.name, f"Unexpected shape {X.shape}")
                )
                continue

            if not np.isfinite(X).all():
                skipped.append(
                    (class_name, path.name, "NaN or infinity in input")
                )
                continue

            # Flag suspicious coordinates; do not silently clip them.
            if np.max(np.abs(X)) > 5:
                print(
                    f"WARNING: extreme coordinate in "
                    f"{class_name}/{path.name}; inspect before training."
                )

            features = extract_features(X)

            if not np.isfinite(features).all():
                skipped.append(
                    (class_name, path.name, "Invalid derived features")
                )
                continue

            save_path = output_path / path.name
            np.save(save_path, features)

            records.append({
                "class": class_name,
                "filename": path.name,
                "feature_path": str(save_path),
                "frames": features.shape[0],
                "features": features.shape[1],
            })

    # Write feature index
    with open(
        OUTPUT_DIR / "feature_index.csv",
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "class",
                "filename",
                "feature_path",
                "frames",
                "features",
            ],
        )
        writer.writeheader()
        writer.writerows(records)

    print("\n--- FEATURE EXTRACTION SUMMARY ---")
    print("Successfully extracted:", len(records))
    print("Skipped:", len(skipped))
    print("Expected feature shape: (30, 12)")
    print("Output directory:", OUTPUT_DIR)

    if skipped:
        print("\nSkipped files:")
        for class_name, filename, reason in skipped:
            print(f"{class_name}/{filename}: {reason}")


if __name__ == "__main__":
    main()

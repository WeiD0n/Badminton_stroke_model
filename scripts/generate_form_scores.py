
from pathlib import Path
import csv
import numpy as np

FEATURE_DIR = Path("dataset/form_features")
OUTPUT_DIR = Path("dataset/form_scores")

CLASSES = ["smash", "drop", "clear"]
FRAMES = 30
FEATURES = 12

# Sequence-level metric indices:
# 0-3: elbow/knee angles
# 4: trunk inclination
# 5-6: shoulder/hip line orientations
# 7-10: joint-angle changes
# 11: trunk-angle change

METRIC_NAMES = [
    "left_elbow_rom",
    "right_elbow_rom",
    "left_knee_rom",
    "right_knee_rom",
    "trunk_median_inclination",
    "trunk_inclination_variability",
    "shoulder_orientation_variability",
    "hip_orientation_variability",
    "left_elbow_change_magnitude",
    "right_elbow_change_magnitude",
    "left_knee_change_magnitude",
    "right_knee_change_magnitude",
    "trunk_change_magnitude",
]

# Metric groups used for transparent component summaries.
JOINT_METRICS = [0, 1, 2, 3, 8, 9, 10, 11]
POSTURE_METRICS = [4, 5, 6, 7]
TIMING_METRICS = [8, 9, 10, 11, 12]


def line_orientation_difference(a, b):
    """Shortest difference between undirected line angles."""
    return (a - b + 90.0) % 180.0 - 90.0


def sequence_metrics(X):
    """Convert one (30, 12) sequence into interpretable metrics."""
    if X.shape != (FRAMES, FEATURES):
        raise ValueError(f"Expected (30, 12), got {X.shape}")

    if not np.isfinite(X).all():
        raise ValueError("Non-finite feature values")

    metrics = []

    # Joint range of motion (ROM)
    for index in [0, 1, 2, 3]:
        metrics.append(float(np.percentile(
            X[:, index], 95
        ) - np.percentile(X[:, index], 5)))

    # Trunk posture level and variability
    metrics.append(float(np.median(X[:, 4])))
    metrics.append(float(
        np.percentile(X[:, 4], 90)
        - np.percentile(X[:, 4], 10)
    ))

    # Variability of shoulder/hip line orientations.
    # Use circular differences to avoid angle-wraparound errors.
    for index in [5, 6]:
        reference = np.median(X[:, index])
        deviations = line_orientation_difference(
            X[:, index], reference
        )
        metrics.append(float(
            np.percentile(deviations, 90)
            - np.percentile(deviations, 10)
        ))

    # Magnitude of movement changes.
    for index in [7, 8, 9, 10, 11]:
        metrics.append(float(np.median(np.abs(X[:, index]))))

    result = np.asarray(metrics, dtype=np.float64)

    if not np.isfinite(result).all():
        raise ValueError("Invalid sequence metrics")

    return result


def robust_reference(metrics_matrix):
    """Compute stroke-specific median and robust scale."""
    median = np.median(metrics_matrix, axis=0)
    q25 = np.percentile(metrics_matrix, 25, axis=0)
    q75 = np.percentile(metrics_matrix, 75, axis=0)

    # IQR / 1.349 approximates standard deviation for
    # normally distributed data. Floor avoids division by zero.
    scale = np.maximum((q75 - q25) / 1.349, 1e-6)

    return median, scale


def deviation_score(metrics, median, scale, indices, max_z=3.0):
    """
    Score closeness to the dataset's typical profile.
    This is NOT a technique-quality score.
    """
    z = np.abs((metrics[indices] - median[indices]) / scale[indices])
    penalty = np.clip(z, 0.0, max_z).mean() / max_z
    return float(100.0 * (1.0 - penalty))


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load sequences first so each stroke has its own reference.
    sequences = {name: [] for name in CLASSES}
    skipped = []

    for class_name in CLASSES:
        folder = FEATURE_DIR / class_name

        if not folder.exists():
            print(f"WARNING: Missing directory: {folder}")
            continue

        for path in sorted(folder.glob("*.npy")):
            # Exclude the known source sequence with extreme coordinates.
            if class_name == "smash" and path.name == "smash_080.npy":
                skipped.append(
                    (class_name, path.name, "Extreme source coordinate")
                )
                continue

            try:
                X = np.load(path, allow_pickle=False)
                metrics = sequence_metrics(X)
                sequences[class_name].append((path.name, metrics))
            except (OSError, ValueError) as exc:
                skipped.append((class_name, path.name, str(exc)))

    # Estimate references using only the available sequences of each stroke.
    records = []

    for class_name in CLASSES:
        items = sequences[class_name]

        if len(items) < 10:
            print(
                f"WARNING: Only {len(items)} {class_name} sequences; "
                "reference statistics may be unstable."
            )

        if not items:
            continue

        matrix = np.vstack([metrics for _, metrics in items])
        median, scale = robust_reference(matrix)

        for filename, metrics in items:
            joint = deviation_score(
                metrics, median, scale, JOINT_METRICS
            )
            posture = deviation_score(
                metrics, median, scale, POSTURE_METRICS
            )
            timing = deviation_score(
                metrics, median, scale, TIMING_METRICS
            )

            # Component weights retained from the prototype design.
            total = (
                0.40 * joint
                + 0.35 * posture
                + 0.25 * timing
            )

            records.append({
                "class": class_name,
                "filename": filename,
                "joint_profile_score": round(joint, 2),
                "posture_profile_score": round(posture, 2),
                "timing_profile_score": round(timing, 2),
                "movement_profile_score": round(total, 2),
            })

    output_csv = OUTPUT_DIR / "form_scores.csv"

    with open(
        output_csv, "w", newline="", encoding="utf-8"
    ) as f:
        fieldnames = [
            "class",
            "filename",
            "joint_profile_score",
            "posture_profile_score",
            "timing_profile_score",
            "movement_profile_score",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print("\n--- REVISED MOVEMENT PROFILE SUMMARY ---")
    print("Sequences scored:", len(records))
    print("Sequences skipped:", len(skipped))
    print("Output:", output_csv)

    if records:
        scores = np.array(
            [r["movement_profile_score"] for r in records]
        )
        print(f"Mean: {scores.mean():.2f}")
        print(f"Median: {np.median(scores):.2f}")
        print(f"Standard deviation: {scores.std():.2f}")
        print(f"Minimum: {scores.min():.2f}")
        print(f"Maximum: {scores.max():.2f}")

        print("\nMean movement-profile score by stroke:")
        for class_name in CLASSES:
            values = [
                r["movement_profile_score"]
                for r in records
                if r["class"] == class_name
            ]
            if values:
                print(
                    f"{class_name}: {np.mean(values):.2f} "
                    f"({len(values)} sequences)"
                )

    if skipped:
        print("\nSkipped files:")
        for class_name, filename, reason in skipped:
            print(f"{class_name}/{filename}: {reason}")


if __name__ == "__main__":
    main()

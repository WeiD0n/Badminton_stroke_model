
from pathlib import Path
import csv
import numpy as np

FEATURE_DIR = Path("dataset/form_features")
OUTPUT_DIR = Path("dataset/form_scores")
OUTPUT_CSV = OUTPUT_DIR / "form_scores_v2.csv"

CLASSES = ["smash", "drop", "clear"]
FRAMES = 30
FEATURES = 12

# Feature indices
LEFT_ELBOW, RIGHT_ELBOW = 0, 1
LEFT_KNEE, RIGHT_KNEE = 2, 3
TRUNK = 4
SHOULDER, HIP = 5, 6
LEFT_ELBOW_CHANGE, RIGHT_ELBOW_CHANGE = 7, 8
LEFT_KNEE_CHANGE, RIGHT_KNEE_CHANGE = 9, 10
TRUNK_CHANGE = 11

# Component weights: proposed prototype design, not validated weights.
WEIGHTS = {
    "joint_movement": 40,
    "posture": 30,
    "temporal_movement": 30,
}

# Only the known corrupted source sequence is excluded.
EXCLUDED = {("smash", "smash_080.npy")}


def line_difference(a, b):
    """Shortest signed difference between undirected line angles."""
    return (a - b + 90.0) % 180.0 - 90.0


def sequence_metrics(X):
    """Extract interpretable movement metrics from one (30, 12) sequence."""
    if X.shape != (FRAMES, FEATURES):
        raise ValueError(f"Expected (30, 12), got {X.shape}")

    if not np.isfinite(X).all():
        raise ValueError("Sequence contains NaN or infinity")

    result = {}

    # Joint range of motion: 95th minus 5th percentile.
    for index, name in [
        (LEFT_ELBOW, "left_elbow_rom"),
        (RIGHT_ELBOW, "right_elbow_rom"),
        (LEFT_KNEE, "left_knee_rom"),
        (RIGHT_KNEE, "right_knee_rom"),
    ]:
        result[name] = float(
            np.percentile(X[:, index], 95)
            - np.percentile(X[:, index], 5)
        )

    # Trunk inclination and variation.
    result["trunk_median"] = float(np.median(X[:, TRUNK]))
    result["trunk_variability"] = float(
        np.percentile(X[:, TRUNK], 90)
        - np.percentile(X[:, TRUNK], 10)
    )

    # Circular variability for shoulder and hip line orientations.
    for index, name in [
        (SHOULDER, "shoulder_orientation_variability"),
        (HIP, "hip_orientation_variability"),
    ]:
        angles = X[:, index]
        radians = np.deg2rad(2.0 * angles)

        mean_angle = np.arctan2(
            np.mean(np.sin(radians)),
            np.mean(np.cos(radians)),
        ) / 2.0

        reference = np.rad2deg(mean_angle)
        deviations = line_difference(angles, reference)

        result[name] = float(
            np.percentile(deviations, 90)
            - np.percentile(deviations, 10)
        )

    # Movement-change magnitudes.
    for index, name in [
        (LEFT_ELBOW_CHANGE, "left_elbow_change"),
        (RIGHT_ELBOW_CHANGE, "right_elbow_change"),
        (LEFT_KNEE_CHANGE, "left_knee_change"),
        (RIGHT_KNEE_CHANGE, "right_knee_change"),
        (TRUNK_CHANGE, "trunk_change"),
    ]:
        changes = np.abs(X[:, index])
        result[f"{name}_median"] = float(np.median(changes))
        result[f"{name}_p95"] = float(np.percentile(changes, 95))

    return result


JOINT_METRICS = [
    "left_elbow_rom",
    "right_elbow_rom",
    "left_knee_rom",
    "right_knee_rom",
]

POSTURE_METRICS = [
    "trunk_median",
    "trunk_variability",
    "shoulder_orientation_variability",
    "hip_orientation_variability",
]

TEMPORAL_METRICS = [
    "left_elbow_change_median",
    "right_elbow_change_median",
    "left_knee_change_median",
    "right_knee_change_median",
    "trunk_change_median",
]


def fit_reference(rows):
    """
    Calculate stroke-specific reference ranges from the available
    professional sequences. P10-P90 defines the typical range.
    """
    metrics = sorted(rows[0]["metrics"].keys())
    reference = {}

    for name in metrics:
        values = np.array(
            [row["metrics"][name] for row in rows],
            dtype=float,
        )

        reference[name] = {
            "p10": float(np.percentile(values, 10)),
            "p50": float(np.percentile(values, 50)),
            "p90": float(np.percentile(values, 90)),
        }

    return reference


def metric_similarity(value, ref):
    """
    100 inside the reference P10-P90 range.
    Outside the range, decrease gradually based on the distance
    relative to the reference range width.

    This measures reference similarity, not technique quality.
    """
    low, high = ref["p10"], ref["p90"]
    width = high - low

    # Avoid unstable scores for nearly constant reference metrics.
    scale = max(width, 1.0)

    if low <= value <= high:
        return 100.0

    if value < low:
        distance = (low - value) / scale
    else:
        distance = (value - high) / scale

    return float(np.clip(100.0 * (1.0 - distance), 0.0, 100.0))


def component_score(metrics, reference, metric_names):
    scores = [
        metric_similarity(metrics[name], reference[name])
        for name in metric_names
    ]
    return float(np.mean(scores))


def feedback_for_component(name, metrics, reference, metric_names):
    outside = []

    for metric in metric_names:
        value = metrics[metric]
        ref = reference[metric]

        if value < ref["p10"]:
            outside.append(f"{metric} is below the reference range")
        elif value > ref["p90"]:
            outside.append(f"{metric} is above the reference range")

    if not outside:
        return f"{name}: measured features are within the reference ranges."

    return f"{name}: " + "; ".join(outside[:3]) + "."


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load and calculate metrics before building references.
    sequences = {name: [] for name in CLASSES}
    skipped = []

    for class_name in CLASSES:
        folder = FEATURE_DIR / class_name

        if not folder.exists():
            print(f"WARNING: Missing directory: {folder}")
            continue

        for path in sorted(folder.glob("*.npy")):
            if (class_name, path.name) in EXCLUDED:
                skipped.append((class_name, path.name, "Excluded extreme source"))
                continue

            try:
                X = np.load(path, allow_pickle=False)
                metrics = sequence_metrics(X)

                sequences[class_name].append({
                    "class": class_name,
                    "filename": path.name,
                    "metrics": metrics,
                })

            except (OSError, ValueError) as exc:
                skipped.append((class_name, path.name, str(exc)))

    records = []

    for class_name in CLASSES:
        rows = sequences[class_name]

        if len(rows) < 10:
            print(
                f"WARNING: Only {len(rows)} {class_name} sequences; "
                "reference ranges may be unstable."
            )

        if not rows:
            continue

        reference = fit_reference(rows)

        for row in rows:
            metrics = row["metrics"]

            joint = component_score(
                metrics, reference, JOINT_METRICS
            )
            posture = component_score(
                metrics, reference, POSTURE_METRICS
            )
            temporal = component_score(
                metrics, reference, TEMPORAL_METRICS
            )

            total = (
                WEIGHTS["joint_movement"] * joint
                + WEIGHTS["posture"] * posture
                + WEIGHTS["temporal_movement"] * temporal
            ) / 100.0

            feedback = " ".join([
                feedback_for_component(
                    "Joint movement", metrics, reference, JOINT_METRICS
                ),
                feedback_for_component(
                    "Posture", metrics, reference, POSTURE_METRICS
                ),
                feedback_for_component(
                    "Temporal movement", metrics, reference, TEMPORAL_METRICS
                ),
            ])

            records.append({
                "class": class_name,
                "filename": row["filename"],
                "joint_profile_score": round(joint, 2),
                "posture_profile_score": round(posture, 2),
                "temporal_profile_score": round(temporal, 2),
                "movement_profile_score": round(total, 2),
                "feedback": feedback,
            })

    fields = [
        "class",
        "filename",
        "joint_profile_score",
        "posture_profile_score",
        "temporal_profile_score",
        "movement_profile_score",
        "feedback",
    ]

    with open(
        OUTPUT_CSV, "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

    print("\n--- POSE MOVEMENT-PROFILE SCORING ---")
    print("Sequences scored:", len(records))
    print("Sequences skipped:", len(skipped))
    print("Output:", OUTPUT_CSV)

    if records:
        scores = np.array(
            [r["movement_profile_score"] for r in records],
            dtype=float,
        )

        print(f"Mean profile-match score: {scores.mean():.2f}")
        print(f"Median profile-match score: {np.median(scores):.2f}")
        print(f"Standard deviation: {scores.std():.2f}")
        print(f"Minimum: {scores.min():.2f}")
        print(f"Maximum: {scores.max():.2f}")

        print("\nMean score by stroke:")
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
        print("\nSkipped sequences:")
        for class_name, filename, reason in skipped:
            print(f"{class_name}/{filename}: {reason}")


if __name__ == "__main__":
    main()

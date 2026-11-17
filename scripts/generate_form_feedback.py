
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FEATURE_DIR = ROOT / "dataset" / "form_features"
OUTPUT_DIR = ROOT / "dataset" / "form_scores"

OUTPUT_FILE = OUTPUT_DIR / "form_feedback.csv"
MAX_FEEDBACK = 3

# Feature indices from extract_form_features.py
LEFT_ELBOW = 0
RIGHT_ELBOW = 1
LEFT_KNEE = 2
RIGHT_KNEE = 3
TRUNK = 4
SHOULDER_LINE = 5
HIP_LINE = 6
LEFT_ELBOW_CHANGE = 7
RIGHT_ELBOW_CHANGE = 8
LEFT_KNEE_CHANGE = 9
RIGHT_KNEE_CHANGE = 10
TRUNK_CHANGE = 11


def circular_variability(values):
    """Measure variation in an undirected line orientation."""
    radians = np.deg2rad(values * 2)
    resultant = np.hypot(
        np.mean(np.cos(radians)),
        np.mean(np.sin(radians)),
    )
    return float(
        np.degrees(
            np.sqrt(max(0, -2 * np.log(max(resultant, 1e-8))))
        ) / 2
    )


def calculate_metrics(features):
    if features.shape != (30, 12):
        raise ValueError(f"Expected (30, 12), received {features.shape}")

    if not np.isfinite(features).all():
        raise ValueError("Sequence contains invalid values")

    def angle_range(index):
        return float(
            np.percentile(features[:, index], 90)
            - np.percentile(features[:, index], 10)
        )

    def median_change(index):
        return float(np.median(np.abs(features[:, index])))

    return {
        "left_elbow_rom": angle_range(LEFT_ELBOW),
        "right_elbow_rom": angle_range(RIGHT_ELBOW),
        "left_knee_rom": angle_range(LEFT_KNEE),
        "right_knee_rom": angle_range(RIGHT_KNEE),
        "trunk_median": float(np.median(features[:, TRUNK])),
        "trunk_variability": angle_range(TRUNK),
        "shoulder_orientation_variability": circular_variability(
            features[:, SHOULDER_LINE]
        ),
        "hip_orientation_variability": circular_variability(
            features[:, HIP_LINE]
        ),
        "left_elbow_change_median": median_change(LEFT_ELBOW_CHANGE),
        "right_elbow_change_median": median_change(RIGHT_ELBOW_CHANGE),
        "left_knee_change_median": median_change(LEFT_KNEE_CHANGE),
        "right_knee_change_median": median_change(RIGHT_KNEE_CHANGE),
        "trunk_change_median": median_change(TRUNK_CHANGE),
    }


# Each rule contains plain-English wording and a category.
RULES = {
    "left_elbow_rom": (
        "Joint movement",
        "Left elbow movement",
        "Your left elbow moved through a smaller range of angles than in most reference videos.",
        "Your left elbow moved through a larger range of angles than in most reference videos.",
        "This describes how much the angle at your left elbow changed during the clip.",
    ),
    "right_elbow_rom": (
        "Joint movement",
        "Right elbow movement",
        "Your right elbow moved through a smaller range of angles than in most reference videos.",
        "Your right elbow moved through a larger range of angles than in most reference videos.",
        "This describes how much the angle at your right elbow changed during the clip.",
    ),
    "left_knee_rom": (
        "Joint movement",
        "Left knee movement",
        "Your left knee bent and straightened through a smaller range than in most reference videos.",
        "Your left knee bent and straightened through a larger range than in most reference videos.",
        "A larger or smaller range is not automatically better; it depends on the shot and movement phase.",
    ),
    "right_knee_rom": (
        "Joint movement",
        "Right knee movement",
        "Your right knee bent and straightened through a smaller range than in most reference videos.",
        "Your right knee bent and straightened through a larger range than in most reference videos.",
        "A larger or smaller range is not automatically better; it depends on the shot and movement phase.",
    ),
    "trunk_median": (
        "Posture alignment",
        "Upper-body position",
        "Your upper body stayed more upright than in most reference videos.",
        "Your upper body leaned farther forward than in most reference videos.",
        "This describes your upper-body position in the video. The appropriate lean depends on the shot and camera angle.",
    ),
    "trunk_variability": (
        "Posture alignment",
        "Upper-body movement",
        "Your upper-body angle changed less throughout the clip than in most reference videos.",
        "Your upper-body angle changed more throughout the clip than in most reference videos.",
        "This measures how much your upper-body angle varied, not whether that variation was correct.",
    ),
    "shoulder_orientation_variability": (
        "Posture alignment",
        "Shoulder-line movement",
        "The angle of your shoulder line changed less than in most reference videos.",
        "The angle of your shoulder line changed more than in most reference videos.",
        "Shoulder rotation can be part of a badminton stroke. This measurement alone cannot judge its quality.",
    ),
    "hip_orientation_variability": (
        "Posture alignment",
        "Hip-line movement",
        "The angle of your hip line changed less than in most reference videos.",
        "The angle of your hip line changed more than in most reference videos.",
        "This describes visible hip-line movement; it does not directly measure three-dimensional hip rotation.",
    ),
    "left_elbow_change_median": (
        "Movement timing",
        "Left elbow movement between frames",
        "Your left elbow angle changed less from frame to frame than in most reference videos.",
        "Your left elbow angle changed more from frame to frame than in most reference videos.",
        "This measures changes between sampled frames, not the actual speed or timing of your stroke.",
    ),
    "right_elbow_change_median": (
        "Movement timing",
        "Right elbow movement between frames",
        "Your right elbow angle changed less from frame to frame than in most reference videos.",
        "Your right elbow angle changed more from frame to frame than in most reference videos.",
        "This measures changes between sampled frames, not the actual speed or timing of your stroke.",
    ),
    "left_knee_change_median": (
        "Movement timing",
        "Left knee movement between frames",
        "Your left knee angle changed less from frame to frame than in most reference videos.",
        "Your left knee angle changed more from frame to frame than in most reference videos.",
        "This describes frame-to-frame angle changes, not whether your knee movement was correctly timed.",
    ),
    "right_knee_change_median": (
        "Movement timing",
        "Right knee movement between frames",
        "Your right knee angle changed less from frame to frame than in most reference videos.",
        "Your right knee angle changed more from frame to frame than in most reference videos.",
        "This describes frame-to-frame angle changes, not whether your knee movement was correctly timed.",
    ),
    "trunk_change_median": (
        "Movement timing",
        "Upper-body movement between frames",
        "Your upper-body angle changed less from frame to frame than in most reference videos.",
        "Your upper-body angle changed more from frame to frame than in most reference videos.",
        "This measures changes between sampled frames, not the timing of your full stroke.",
    ),
}


def main():
    if not FEATURE_DIR.exists():
        raise FileNotFoundError(f"Missing directory: {FEATURE_DIR}")

    records = []

    for stroke_dir in sorted(FEATURE_DIR.iterdir()):
        if not stroke_dir.is_dir():
            continue

        stroke = stroke_dir.name.lower()

        for path in sorted(stroke_dir.glob("*.npy")):
            if stroke == "smash" and path.name == "smash_080.npy":
                continue

            try:
                features = np.load(path, allow_pickle=False)
                metrics = calculate_metrics(features)
                records.append({
                    "filename": path.name,
                    "class": stroke,
                    **metrics,
                })
            except (ValueError, OSError) as exc:
                print(f"Skipped {path.name}: {exc}")

    df = pd.DataFrame(records)

    if df.empty:
        raise ValueError("No valid feature sequences were found.")

    # Reference ranges are calculated separately for each stroke.
    references = {}

    for stroke, group in df.groupby("class"):
        references[stroke] = {}

        for metric in RULES:
            values = group[metric]
            references[stroke][metric] = {
                "low": float(values.quantile(0.10)),
                "high": float(values.quantile(0.90)),
            }

    output_rows = []

    for _, row in df.iterrows():
        candidates = []

        for metric, rule in RULES.items():
            value = float(row[metric])
            ref = references[row["class"]][metric]
            low, high = ref["low"], ref["high"]

            if low <= value <= high:
                continue

            # Rank observations by how far they fall outside the
            # reference interval, scaled by its width.
            width = max(high - low, 1e-6)

            if value < low:
                deviation = (low - value) / width
                message = rule[2]
            else:
                deviation = (value - high) / width
                message = rule[3]

            candidates.append({
                "metric": metric,
                "category": rule[0],
                "title": rule[1],
                "message": message,
                "explanation": rule[4],
                "measured_value": round(value, 2),
                "reference_range": f"{low:.2f}–{high:.2f}°",
                "deviation": deviation,
            })
        # Prioritise the strongest difference in each category first.
        candidates.sort(
            key=lambda item: item["deviation"],
            reverse=True,
        )

        selected = []
        used_categories = set()

        # First pass: choose different categories where available.
        for item in candidates:
            if item["category"] not in used_categories:
                selected.append(item)
                used_categories.add(item["category"])

            if len(selected) == MAX_FEEDBACK:
                break

        # Second pass: fill any remaining spaces with the next strongest findings.
        if len(selected) < MAX_FEEDBACK:
            selected_metrics = {item["metric"] for item in selected}

            for item in candidates:
                if item["metric"] not in selected_metrics:
                    selected.append(item)
                    selected_metrics.add(item["metric"])

                if len(selected) == MAX_FEEDBACK:
                    break


        output = {
            "filename": row["filename"],
            "class": row["class"],
            "feedback_count": len(selected),
            "feedback": " ".join(item["message"] for item in selected),
            "interpretation_note": (
                "These observations compare measurements with collected "
                "videos; they are not expert-validated technique grades."
            ),
        }

        # Structured columns are easier for the React frontend to consume.
        for position in range(1, MAX_FEEDBACK + 1):
            item = selected[position - 1] if position <= len(selected) else {}

            for field in (
                "category",
                "title",
                "message",
                "measured_value",
                "reference_range",
                "explanation",
            ):
                output[f"feedback_{position}_{field}"] = item.get(field, "")

        output_rows.append(output)

    result = pd.DataFrame(output_rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print("\n--- BEGINNER-FRIENDLY FORM FEEDBACK ---")
    print(f"Sequences processed: {len(result)}")
    print(f"Output: {OUTPUT_FILE}")

    for _, row in result.head(5).iterrows():
        print(f"\n{row['filename']} ({row['class']})")

        if row["feedback_count"] == 0:
            print("No measurements fell outside the reference ranges.")
        else:
            for position in range(1, int(row["feedback_count"]) + 1):
                print(
                    f"- [{row[f'feedback_{position}_category']}] "
                    f"{row[f'feedback_{position}_title']}: "
                    f"{row[f'feedback_{position}_message']}"
                )
                print(
                    f"  Measured: {row[f'feedback_{position}_measured_value']}°; "
                    f"reference: {row[f'feedback_{position}_reference_range']}"
                )


if __name__ == "__main__":
    main()

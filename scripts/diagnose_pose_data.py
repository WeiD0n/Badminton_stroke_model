
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split

# ============================================================
# CONFIGURATION
# ============================================================

SEQUENCE_DIR = Path("dataset/sequences_tracked")
VISIBILITY_DIR = Path("dataset/visibility_tracked")
OUTPUT_DIR = Path("results/pose_diagnostics")

CLASSES = ["smash", "drop", "clear"]
FRAMES = 30
LANDMARKS = 33
COORDS = 3
FEATURES = LANDMARKS * COORDS
SEED = 42

# MediaPipe Pose landmark names, in standard landmark order.
LANDMARK_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index"
]

# ============================================================
# LOAD FILES
# ============================================================

X_list = []
y_list = []
names = []
V_list = []
visibility_available = True

print("=" * 70)
print("POSE DATA DIAGNOSTICS")
print("=" * 70)

for class_name in CLASSES:
    folder = SEQUENCE_DIR / class_name

    if not folder.exists():
        raise FileNotFoundError(f"Missing folder: {folder}")

    files = sorted(folder.glob("*.npy"))
    print(f"\n{class_name.upper()}: {len(files)} sequence files")

    for path in files:
        sequence = np.load(path)

        if sequence.shape != (FRAMES, FEATURES):
            print(
                f"ERROR: {path.name} shape {sequence.shape}; "
                f"expected {(FRAMES, FEATURES)}"
            )
            continue

        if not np.isfinite(sequence).all():
            print(f"ERROR: Non-finite values in {path}")
            continue

        X_list.append(sequence.astype(np.float32))
        y_list.append(class_name)
        names.append(path.name)

        visibility_path = (
            VISIBILITY_DIR / class_name / path.name
        )

        if visibility_path.exists():
            visibility = np.load(visibility_path)

            if visibility.shape != (FRAMES, LANDMARKS):
                print(
                    f"WARNING: {visibility_path.name} visibility "
                    f"shape is {visibility.shape}"
                )
                V_list.append(None)
            else:
                V_list.append(visibility.astype(np.float32))
        else:
            visibility_available = False
            V_list.append(None)

if len(X_list) == 0:
    raise RuntimeError("No valid sequences were loaded.")

X = np.stack(X_list)
labels = np.array(y_list)
names = np.array(names)

print("\nLoaded sequences:", len(X))
print("X shape:", X.shape)
print("Expected shape per sequence:", (FRAMES, FEATURES))

# ============================================================
# BASIC DATA CHECKS
# ============================================================

print("\n" + "=" * 70)
print("1. BASIC FEATURE CHECKS")
print("=" * 70)

print("Overall minimum:", float(X.min()))
print("Overall maximum:", float(X.max()))
print("Overall mean:", float(X.mean()))
print("Overall standard deviation:", float(X.std()))
print("All values finite:", bool(np.isfinite(X).all()))

for class_name in CLASSES:
    class_X = X[labels == class_name]

    if len(class_X) == 0:
        continue

    print(
        f"{class_name:>6}: n={len(class_X)}, "
        f"mean={class_X.mean():.4f}, "
        f"std={class_X.std():.4f}, "
        f"min={class_X.min():.4f}, "
        f"max={class_X.max():.4f}"
    )

# Check features that barely change across all frames/videos.
feature_std = X.reshape(-1, FEATURES).std(axis=0)
constant_features = np.where(feature_std < 1e-6)[0]

print("\nNear-constant features:", len(constant_features))
if len(constant_features):
    print("Feature indices:", constant_features.tolist())

# ============================================================
# PER-SEQUENCE CHECKS
# ============================================================

print("\n" + "=" * 70)
print("2. PER-SEQUENCE CHECKS")
print("=" * 70)

for i, sequence in enumerate(X):
    frame_std = sequence.std(axis=1)
    frame_means = sequence.mean(axis=1)

    low_variation_frames = int(np.sum(frame_std < 1e-6))
    repeated_frames = int(
        np.sum(np.all(np.isclose(sequence[1:], sequence[:-1]), axis=1))
    )

    print(
        f"{names[i]:15} label={labels[i]:6} "
        f"mean={sequence.mean(): .4f} "
        f"std={sequence.std(): .4f} "
        f"frame_mean_range="
        f"{frame_means.min():.3f}..{frame_means.max():.3f} "
        f"low_variation_frames={low_variation_frames} "
        f"identical_adjacent_frames={repeated_frames}"
    )

# ============================================================
# CLASS FEATURE DISTRIBUTIONS
# ============================================================

print("\n" + "=" * 70)
print("3. CLASS FEATURE DISTRIBUTIONS")
print("=" * 70)

class_feature_stats = {}

for class_name in CLASSES:
    class_X = X[labels == class_name]

    if len(class_X) == 0:
        continue

    # Aggregate across videos and frames for each x/y/z feature.
    feature_means = class_X.reshape(-1, FEATURES).mean(axis=0)
    feature_stds = class_X.reshape(-1, FEATURES).std(axis=0)

    class_feature_stats[class_name] = {
        "mean": feature_means,
        "std": feature_stds
    }

    print(f"\n{class_name.upper()}:")
    print("  Average feature mean:", float(feature_means.mean()))
    print("  Average feature std:", float(feature_stds.mean()))

print("\nLargest differences in average feature means between classes:")
pairs = [
    ("smash", "drop"),
    ("smash", "clear"),
    ("drop", "clear")
]

for a, b in pairs:
    if a not in class_feature_stats or b not in class_feature_stats:
        continue

    diff = np.abs(
        class_feature_stats[a]["mean"] -
        class_feature_stats[b]["mean"]
    )

    top = np.argsort(diff)[-5:][::-1]

    print(f"\n{a} vs {b}:")
    for index in top:
        landmark_index = index // COORDS
        coord_name = ["x", "y", "z"][index % COORDS]

        print(
            f"  feature {index:2} "
            f"({LANDMARK_NAMES[landmark_index]}.{coord_name}): "
            f"mean difference={diff[index]:.5f}"
        )

# ============================================================
# VISIBILITY CHECKS
# ============================================================

print("\n" + "=" * 70)
print("4. LANDMARK VISIBILITY CHECKS")
print("=" * 70)

if not visibility_available or any(v is None for v in V_list):
    print(
        "Some visibility files are missing or invalid. "
        "Check dataset/visibility_tracked/."
    )
else:
    V = np.stack(V_list)

    print("Visibility shape:", V.shape)
    print("Visibility min/max:", float(V.min()), float(V.max()))

    # Values below this threshold are treated as low visibility.
    threshold = 0.35

    for class_name in CLASSES:
        mask = labels == class_name
        class_V = V[mask]

        if len(class_V) == 0:
            continue

        low_fraction = float(np.mean(class_V < threshold))

        print(
            f"\n{class_name.upper()}: "
            f"mean visibility={class_V.mean():.3f}, "
            f"fraction below {threshold}={low_fraction:.1%}"
        )

        # Landmarks with the highest fraction of low-visibility frames.
        landmark_low = np.mean(class_V < threshold, axis=(0, 1))
        worst = np.argsort(landmark_low)[-5:][::-1]

        print("  Landmarks most often below threshold:")
        for landmark_index in worst:
            print(
                f"    {LANDMARK_NAMES[landmark_index]}: "
                f"{landmark_low[landmark_index]:.1%}"
            )

# ============================================================
# RECREATE THE SAME TRAIN/TEST SPLIT
# ============================================================

print("\n" + "=" * 70)
print("5. TRAIN / TEST FEATURE COMPARISON")
print("=" * 70)

label_to_id = {"clear": 0, "drop": 1, "smash": 2}
y = np.array([label_to_id[label] for label in labels])
indices = np.arange(len(X))

train_idx, test_idx = train_test_split(
    indices,
    test_size=0.40,
    random_state=SEED,
    stratify=y
)

print("Training files:")
for i in train_idx:
    print(f"  {names[i]} -> {labels[i]}")

print("\nTest files:")
for i in test_idx:
    print(f"  {names[i]} -> {labels[i]}")

train_X = X[train_idx]
test_X = X[test_idx]

print("\nTraining feature mean/std:",
      float(train_X.mean()), float(train_X.std()))
print("Test feature mean/std:",
      float(test_X.mean()), float(test_X.std()))

print(
    "Absolute difference in overall means:",
    float(abs(train_X.mean() - test_X.mean()))
)

print(
    "Absolute difference in overall standard deviations:",
    float(abs(train_X.std() - test_X.std()))
)

# ============================================================
# SAVE A TEXT REPORT
# ============================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
report_path = OUTPUT_DIR / "diagnostic_summary.txt"

with open(report_path, "w", encoding="utf-8") as f:
    f.write("POSE DATA DIAGNOSTICS\n")
    f.write("=" * 60 + "\n")
    f.write(f"Sequences loaded: {len(X)}\n")
    f.write(f"Shape: {X.shape}\n")
    f.write(f"Overall mean: {X.mean():.6f}\n")
    f.write(f"Overall std: {X.std():.6f}\n")
    f.write(f"Near-constant features: {len(constant_features)}\n")
    f.write(f"Visibility files complete: {visibility_available}\n")
    f.write("\nClass counts:\n")

    for class_name in CLASSES:
        f.write(f"{class_name}: {int(np.sum(labels == class_name))}\n")

    f.write("\nTraining files:\n")
    for i in train_idx:
        f.write(f"{names[i]} -> {labels[i]}\n")

    f.write("\nTest files:\n")
    for i in test_idx:
        f.write(f"{names[i]} -> {labels[i]}\n")

print("\nSaved brief report to:", report_path.resolve())
print("\nDiagnostics complete. No source data was modified.")

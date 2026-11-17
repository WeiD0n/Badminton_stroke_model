
from pathlib import Path
import numpy as np

seq_root = Path("dataset/sequences_tracked")
vis_root = Path("dataset/visibility_tracked")

errors = 0

for class_name in ["smash", "drop", "clear"]:
    for seq_path in sorted((seq_root / class_name).glob("*.npy")):
        vis_path = vis_root / class_name / f"{seq_path.stem}_visibility.npy"

        if not vis_path.exists():
            print(f"MISSING: {vis_path}")
            errors += 1
            continue

        seq = np.load(seq_path)
        vis = np.load(vis_path)

        valid = (
            seq.shape == (30, 99)
            and vis.shape == (30, 33)
            and np.isfinite(vis).all()
            and vis.min() >= 0
            and vis.max() <= 1
        )

        print(
            f"{class_name}/{seq_path.name}: "
            f"sequence={seq.shape}, visibility={vis.shape}, "
            f"range=({vis.min():.3f}, {vis.max():.3f}), "
            f"valid={valid}"
        )

        if not valid:
            errors += 1

print(f"\nTotal errors: {errors}")

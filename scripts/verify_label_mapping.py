
from pathlib import Path
import numpy as np

DATA_DIR = Path("dataset/sequences_tracked")
CLASSES = ["smash", "drop", "clear"]

print("=" * 60)
print("VERIFYING POSE SEQUENCE LABEL MAPPING")
print("=" * 60)

total = 0
errors = 0

for class_name in CLASSES:
    class_dir = DATA_DIR / class_name

    if not class_dir.exists():
        print(f"\nERROR: Folder not found: {class_dir}")
        errors += 1
        continue

    files = sorted(class_dir.glob("*.npy"))

    print(f"\nExpected class: {class_name.upper()}")
    print(f"Folder: {class_dir}")
    print(f"Files found: {len(files)}")

    for file_path in files:
        total += 1

        try:
            sequence = np.load(file_path)

            valid_shape = sequence.shape == (30, 99)
            finite_values = np.isfinite(sequence).all()
            correct_folder = file_path.parent.name == class_name

            if not valid_shape or not finite_values or not correct_folder:
                errors += 1
                print(f"  ERROR: {file_path.name}")
                print(f"    Shape: {sequence.shape}")
                print(f"    Finite values: {finite_values}")
                print(f"    Folder matches expected class: {correct_folder}")
            else:
                print(f"  OK: {file_path.name} -> {class_name}")

        except Exception as exc:
            errors += 1
            print(f"  ERROR loading {file_path.name}: {exc}")

print("\n" + "=" * 60)
print("VERIFICATION SUMMARY")
print("=" * 60)
print(f"Total files checked: {total}")
print(f"Errors found: {errors}")

if total == 15 and errors == 0:
    print("PASS: All 15 files have valid shapes and folder labels.")
elif errors == 0:
    print("PASS: No errors found in the files that were checked.")
else:
    print("FAIL: Investigate the errors above.")

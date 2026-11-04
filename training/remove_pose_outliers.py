
from pathlib import Path
import shutil

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SEQUENCE_FOLDER = PROJECT_ROOT / "dataset" / "sequences" / "smash"
VISIBILITY_FOLDER = PROJECT_ROOT / "dataset" / "visibility" / "smash"

REJECTED_SEQUENCE_FOLDER = (
    PROJECT_ROOT / "dataset" / "rejected_sequences" / "smash"
)

REJECTED_VISIBILITY_FOLDER = (
    PROJECT_ROOT / "dataset" / "rejected_visibility" / "smash"
)

OUTLIERS = [
    "smash_592",
    "smash_080",
]


def main():
    REJECTED_SEQUENCE_FOLDER.mkdir(
        parents=True, exist_ok=True
    )

    REJECTED_VISIBILITY_FOLDER.mkdir(
        parents=True, exist_ok=True
    )

    for name in OUTLIERS:
        sequence_file = SEQUENCE_FOLDER / f"{name}.npy"
        visibility_file = (
            VISIBILITY_FOLDER / f"{name}_visibility.npy"
        )

        print(f"\nProcessing {name}")

        if sequence_file.exists():
            shutil.move(
                str(sequence_file),
                str(REJECTED_SEQUENCE_FOLDER / sequence_file.name)
            )
            print("Sequence quarantined.")
        else:
            print("Sequence file not found.")

        if visibility_file.exists():
            shutil.move(
                str(visibility_file),
                str(REJECTED_VISIBILITY_FOLDER / visibility_file.name)
            )
            print("Visibility file quarantined.")
        else:
            print("Visibility file not found.")

    print("\nOutlier quarantine completed.")


if __name__ == "__main__":
    main()

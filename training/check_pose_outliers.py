
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEQUENCE_DIR = PROJECT_ROOT / "dataset" / "sequences" / "smash"

FILES_AND_FRAMES = {
    "smash_592.npy": [12, 13, 14, 15, 16, 17],
    "smash_080.npy": [27, 28, 29],
    "smash_405.npy": [22, 23, 24, 25],
}

NUM_LANDMARKS = 33


def inspect_frames(filename, frame_indices):
    path = SEQUENCE_DIR / filename
    X = np.load(path, allow_pickle=False)
    pose = X.reshape(30, NUM_LANDMARKS, 3)

    print("\n" + "=" * 75)
    print(f"FILE: {filename}")
    print("=" * 75)

    for frame_index in frame_indices:
        frame = pose[frame_index]

        print(f"\nFRAME {frame_index}")
        print("-" * 60)

        for landmark_index in range(NUM_LANDMARKS):
            x, y, z = frame[landmark_index]

            print(
                f"Landmark {landmark_index:02d}: "
                f"X={x:10.4f} "
                f"Y={y:10.4f} "
                f"Z={z:10.4f}"
            )


def main():
    for filename, frames in FILES_AND_FRAMES.items():
        inspect_frames(filename, frames)


if __name__ == "__main__":
    main()

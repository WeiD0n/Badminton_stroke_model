import os
import numpy as np

labels = ["smash", "drop", "clear"]

TARGET_FRAMES = 30

for label in labels:

    folder = f"dataset/sequences/{label}"

    for file in os.listdir(folder):

        path = os.path.join(folder, file)

        data = np.load(path)

        # Skip broken files
        if len(data.shape) != 2:
            continue

        num_frames = data.shape[0]

        indices = np.linspace(
            0,
            num_frames - 1,
            TARGET_FRAMES,
            dtype=int
        )

        data_resampled = data[indices]

        np.save(path, data_resampled)

print("Done!")
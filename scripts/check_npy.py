import numpy as np
import os

labels = ["smash", "drop", "clear"]

for label in labels:

    folder = f"dataset/sequences/{label}"

    lengths = []

    for file in os.listdir(folder):

        data = np.load(os.path.join(folder, file))

        if len(data.shape) == 2:
            lengths.append(data.shape[0])

    print(label)
    print("min =", min(lengths))
    print("max =", max(lengths))
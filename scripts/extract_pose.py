import cv2
import mediapipe as mp
import numpy as np
import os

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

input_folder = "dataset/raw_videos"
output_folder = "dataset/sequences"

labels = ["smash", "drop", "clear"]

for label in labels:

    input_path = os.path.join(input_folder, label)
    output_path = os.path.join(output_folder, label)

    for video_file in os.listdir(input_path):

        if not video_file.endswith(".mp4"):
            continue

        video_path = os.path.join(input_path, video_file)

        cap = cv2.VideoCapture(video_path)

        sequence = []

        while cap.isOpened():

            success, frame = cap.read()

            if not success:
                break

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            if results.pose_landmarks:

                landmarks = []

                # Hip center
                left_hip = results.pose_landmarks.landmark[23]
                right_hip = results.pose_landmarks.landmark[24]

                hip_x = (left_hip.x + right_hip.x) / 2
                hip_y = (left_hip.y + right_hip.y) / 2
                hip_z = (left_hip.z + right_hip.z) / 2

                for lm in results.pose_landmarks.landmark:

                    x = lm.x - hip_x
                    y = lm.y - hip_y
                    z = lm.z - hip_z

                    landmarks.extend([x, y, z])

                sequence.append(landmarks)

        cap.release()

        sequence = np.array(sequence)

        filename = os.path.splitext(video_file)[0] + ".npy"

        np.save(
            os.path.join(output_path, filename),
            sequence
        )

        print("Saved:", filename)

print("Finished!")
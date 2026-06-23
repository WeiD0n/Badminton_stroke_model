import cv2
import mediapipe as mp
import csv
import os

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils

LABEL = "smash"  # change later
DATA_PATH = "dataset/data.csv"

# create file if not exists
if not os.path.exists("dataset"):
    os.makedirs("dataset")

if not os.path.isfile(DATA_PATH):
    with open(DATA_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["label"]
        for i in range(33):
            header += [f"x{i}", f"y{i}", f"z{i}"]
        writer.writerow(header)

cap = cv2.VideoCapture(0)

print("Press S to SAVE frame | Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(img_rgb)

    landmarks = []

    if result.pose_landmarks:
        mp_draw.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        for lm in result.pose_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])

    cv2.imshow("Collect Data", frame)

    key = cv2.waitKey(1)

    # SAVE DATA
    if key == ord('s') and len(landmarks) == 99:
        with open(DATA_PATH, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([LABEL] + landmarks)
        print("Saved frame")

    # QUIT
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
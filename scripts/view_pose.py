import cv2
import mediapipe as mp
import os


# ==============================
# MediaPipe Pose
# ==============================

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# ==============================
# Folder settings
# ==============================

input_folder = "dataset/raw_videos/clear"
output_folder = "dataset/pose_previews/clear"

os.makedirs(output_folder, exist_ok=True)


# ==============================
# Process 50 Clear videos
# 001.mp4 → 050.mp4
# ==============================

results_summary = []

for number in range(1, 51):

    video_name = f"{number:03d}.mp4"

    video_path = os.path.join(
        input_folder,
        video_name
    )

    output_name = f"{number:03d}_pose.mp4"

    output_path = os.path.join(
        output_folder,
        output_name
    )

    print("\n==============================")
    print(f"Processing: {video_name}")
    print("==============================")


    # ==============================
    # Check file exists
    # ==============================

    if not os.path.exists(video_path):

        print("WARNING: File not found!")

        continue


    # ==============================
    # Open video
    # ==============================

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():

        print("ERROR: Could not open video.")

        continue


    # ==============================
    # Video information
    # ==============================

    fps = cap.get(cv2.CAP_PROP_FPS)

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )


    # ==============================
    # Create output video
    # ==============================

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    out = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (width, height)
    )


    # ==============================
    # Process frames
    # ==============================

    frame_number = 0
    detected_frames = 0
    total_frames = 0

    while cap.isOpened():

        success, frame = cap.read()

        if not success:
            break

        total_frames += 1


        # Convert BGR → RGB
        image_rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        # Pose detection
        results = pose.process(
            image_rgb
        )


        # Draw pose
        if results.pose_landmarks:

            detected_frames += 1

            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )


        # Add frame number
        cv2.putText(
            frame,
            f"Frame: {frame_number}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )


        # Write frame
        out.write(frame)


        # Show preview
        cv2.imshow(
            "MediaPipe Pose Preview - CLEAR",
            frame
        )


        frame_number += 1


        # Press Q to stop current video
        if cv2.waitKey(1) & 0xFF == ord("q"):

            print("Stopped by user.")

            break


    # ==============================
    # Cleanup current video
    # ==============================

    cap.release()
    out.release()

    cv2.destroyAllWindows()


    # ==============================
    # Calculate detection rate
    # ==============================

    if total_frames > 0:

        detection_rate = (
            detected_frames / total_frames
        ) * 100

    else:

        detection_rate = 0


    # ==============================
    # Save result
    # ==============================

    results_summary.append(
        (
            video_name,
            total_frames,
            detected_frames,
            detection_rate
        )
    )


    print(
        f"Total frames: {total_frames}"
    )

    print(
        f"Detected frames: {detected_frames}"
    )

    print(
        f"Detection rate: {detection_rate:.2f}%"
    )

    print(
        f"Preview saved to: {output_path}"
    )


# ==============================
# Close MediaPipe
# ==============================

pose.close()


# ==============================
# Final summary
# ==============================

print("\n\n========================================")
print("CLEAR DATASET POSE EXTRACTION SUMMARY")
print("========================================")

print(
    f"{'Video':<12}"
    f"{'Frames':<12}"
    f"{'Detected':<12}"
    f"{'Detection %'}"
)

print("----------------------------------------")


for (
    video_name,
    total_frames,
    detected_frames,
    detection_rate
) in results_summary:

    print(
        f"{video_name:<12}"
        f"{total_frames:<12}"
        f"{detected_frames:<12}"
        f"{detection_rate:.2f}%"
    )


print("\nFinished!")
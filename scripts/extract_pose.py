import cv2
import mediapipe as mp
import numpy as np
import os

# ============================================================
# MEDIAPIPE
# ============================================================

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ============================================================
# FOLDERS
# ============================================================

input_folder = "dataset/raw_videos"
output_folder = "dataset/sequences"
preview_folder = "dataset/pose_previews"

# Only Clear for now
labels = ["clear"]

# ============================================================
# SETTINGS
# ============================================================

# Size of the cropped player image sent to MediaPipe
POSE_WIDTH = 512
POSE_HEIGHT = 512

# ============================================================
# TRACKER
# ============================================================

def create_tracker():

    # OpenCV 5 / newer versions
    if hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create()

    # Older OpenCV versions
    if hasattr(cv2, "legacy") and hasattr(
        cv2.legacy,
        "TrackerCSRT_create"
    ):
        return cv2.legacy.TrackerCSRT_create()

    raise RuntimeError(
        "CSRT tracker is not available in this OpenCV installation."
    )


# ============================================================
# PROCESS
# ============================================================

for label in labels:

    input_path = os.path.join(
        input_folder,
        label
    )

    output_path = os.path.join(
        output_folder,
        label
    )

    preview_path = os.path.join(
        preview_folder,
        label
    )

    os.makedirs(
        output_path,
        exist_ok=True
    )

    os.makedirs(
        preview_path,
        exist_ok=True
    )

    video_files = sorted(
        [
            f for f in os.listdir(input_path)
            if f.lower().endswith(
                (".mp4", ".avi", ".mov", ".mkv")
            )
        ]
    )

    print(
        f"\nFound {len(video_files)} videos "
        f"in {label}"
    )

    for video_file in video_files:

        video_path = os.path.join(
            input_path,
            video_file
        )

        print("\n================================")
        print("Processing:", video_file)
        print("================================")

        cap = cv2.VideoCapture(
            video_path
        )

        if not cap.isOpened():

            print(
                "ERROR: Could not open video."
            )

            continue

        # ----------------------------------------------------
        # VIDEO INFO
        # ----------------------------------------------------

        fps = cap.get(
            cv2.CAP_PROP_FPS
        )

        width = int(
            cap.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        )

        height = int(
            cap.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        )

        total_video_frames = int(
            cap.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )

        print(
            f"Resolution: {width} x {height}"
        )

        print(
            f"FPS: {fps:.2f}"
        )

        print(
            f"Total frames: "
            f"{total_video_frames}"
        )

        # ----------------------------------------------------
        # READ FIRST FRAME
        # ----------------------------------------------------

        success, first_frame = cap.read()

        if not success:

            print(
                "ERROR: Could not read first frame."
            )

            cap.release()
            continue

        # ----------------------------------------------------
        # SELECT TARGET PLAYER
        # ----------------------------------------------------

        print(
            "\nSelect the TARGET PLAYER "
            "with your mouse."
        )

        print(
            "Draw a box around the entire player."
        )

        print(
            "Press ENTER or SPACE when finished."
        )

        print(
            "Press C to cancel."
        )

        roi = cv2.selectROI(
            "Select Target Player",
            first_frame,
            fromCenter=False,
            showCrosshair=True
        )

        cv2.destroyWindow(
            "Select Target Player"
        )

        x, y, w, h = [
            int(v) for v in roi
        ]

        if w == 0 or h == 0:

            print(
                "ROI cancelled."
            )

            cap.release()
            continue

        print(
            f"Selected ROI: "
            f"x={x}, y={y}, "
            f"w={w}, h={h}"
        )

        # ----------------------------------------------------
        # CREATE TRACKER
        # ----------------------------------------------------

        tracker = create_tracker()

        tracker.init(
            first_frame,
            (x, y, w, h)
        )

        # ----------------------------------------------------
        # OUTPUT PREVIEW
        # ----------------------------------------------------

        preview_filename = (
            os.path.splitext(video_file)[0]
            + "_tracking_pose.mp4"
        )

        preview_file = os.path.join(
            preview_path,
            preview_filename
        )

        fourcc = cv2.VideoWriter_fourcc(
            *"mp4v"
        )

        out = cv2.VideoWriter(
            preview_file,
            fourcc,
            fps,
            (width, height)
        )

        # ----------------------------------------------------
        # DATA
        # ----------------------------------------------------

        sequence = []

        total_frames = 1
        tracked_frames = 1
        detected_frames = 0

        # Process the first frame as well
        current_frame = first_frame

        # ====================================================
        # FRAME LOOP
        # ====================================================

        while True:

            # ------------------------------------------------
            # TRACK PLAYER
            # ------------------------------------------------

            if total_frames > 1:

                success, current_frame = cap.read()

                if not success:
                    break

                total_frames += 1

                tracking_success, bbox = tracker.update(
                    current_frame
                )

            else:

                tracking_success = True
                bbox = (x, y, w, h)

            preview_frame = (
                current_frame.copy()
            )

            # ------------------------------------------------
            # TRACKING SUCCESS
            # ------------------------------------------------

            if tracking_success:

                tracked_frames += (
                    1 if total_frames > 1 else 0
                )

                tx, ty, tw, th = [
                    int(v) for v in bbox
                ]

                # Keep coordinates inside frame
                tx = max(0, tx)
                ty = max(0, ty)

                tx2 = min(
                    width,
                    tx + tw
                )

                ty2 = min(
                    height,
                    ty + th
                )

                # ------------------------------------------------
                # DRAW TRACKING BOX
                # ------------------------------------------------

                cv2.rectangle(
                    preview_frame,
                    (tx, ty),
                    (tx2, ty2),
                    (0, 255, 0),
                    3
                )

                cv2.putText(
                    preview_frame,
                    "TRACKING TARGET",
                    (tx, max(30, ty - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

                # ------------------------------------------------
                # CROP TARGET PLAYER
                # ------------------------------------------------

                player_crop = current_frame[
                    ty:ty2,
                    tx:tx2
                ]

                if (
                    player_crop.size > 0
                    and player_crop.shape[0] > 20
                    and player_crop.shape[1] > 20
                ):

                    # Resize for MediaPipe
                    player_crop = cv2.resize(
                        player_crop,
                        (
                            POSE_WIDTH,
                            POSE_HEIGHT
                        ),
                        interpolation=cv2.INTER_CUBIC
                    )

                    # ------------------------------------------------
                    # MEDIAPIPE
                    # ------------------------------------------------

                    image_rgb = cv2.cvtColor(
                        player_crop,
                        cv2.COLOR_BGR2RGB
                    )

                    image_rgb.flags.writeable = False

                    results = pose.process(
                        image_rgb
                    )

                    image_rgb.flags.writeable = True

                    # ------------------------------------------------
                    # POSE DETECTED
                    # ------------------------------------------------

                    if results.pose_landmarks:

                        detected_frames += 1

                        landmarks = []

                        left_hip = (
                            results
                            .pose_landmarks
                            .landmark[23]
                        )

                        right_hip = (
                            results
                            .pose_landmarks
                            .landmark[24]
                        )

                        # Hip centre
                        hip_x = (
                            left_hip.x
                            + right_hip.x
                        ) / 2

                        hip_y = (
                            left_hip.y
                            + right_hip.y
                        ) / 2

                        hip_z = (
                            left_hip.z
                            + right_hip.z
                        ) / 2

                        # 33 landmarks × 3
                        for lm in (
                            results
                            .pose_landmarks
                            .landmark
                        ):

                            px = lm.x - hip_x
                            py = lm.y - hip_y
                            pz = lm.z - hip_z

                            landmarks.extend(
                                [px, py, pz]
                            )

                        sequence.append(
                            landmarks
                        )

                        # Draw pose on cropped player
                        mp_drawing.draw_landmarks(
                            player_crop,
                            results.pose_landmarks,
                            mp_pose.POSE_CONNECTIONS
                        )

                        # ------------------------------------------------
                        # PUT POSE CROP IN PREVIEW
                        # ------------------------------------------------

                        # Resize pose crop for display
                        display_w = min(
                            300,
                            width // 3
                        )

                        display_h = int(
                            display_w
                            * POSE_HEIGHT
                            / POSE_WIDTH
                        )

                        pose_display = cv2.resize(
                            player_crop,
                            (
                                display_w,
                                display_h
                            )
                        )

                        # Put pose preview at top-left
                        preview_frame[
                            10:
                            10 + display_h,
                            10:
                            10 + display_w
                        ] = pose_display

                        cv2.putText(
                            preview_frame,
                            "POSE",
                            (20, 35),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 0),
                            2
                        )

            else:

                # ------------------------------------------------
                # TRACKING LOST
                # ------------------------------------------------

                cv2.putText(
                    preview_frame,
                    "TRACKING LOST",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 0, 255),
                    3
                )

            # ------------------------------------------------
            # FRAME NUMBER
            # ------------------------------------------------

            cv2.putText(
                preview_frame,
                f"Frame: {total_frames}",
                (30, height - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            out.write(
                preview_frame
            )

        # ====================================================
        # FINISH VIDEO
        # ====================================================

        cap.release()
        out.release()

        # ----------------------------------------------------
        # SAVE SEQUENCE
        # ----------------------------------------------------

        sequence = np.array(
            sequence,
            dtype=np.float32
        )

        filename = (
            os.path.splitext(video_file)[0]
            + ".npy"
        )

        output_file = os.path.join(
            output_path,
            filename
        )

        if len(sequence) > 0:

            np.save(
                output_file,
                sequence
            )

            detection_rate = (
                detected_frames
                / max(1, total_frames)
                * 100
            )

            tracking_rate = (
                tracked_frames
                / max(1, total_frames)
                * 100
            )

            print("\nRESULT")
            print("--------------------------------")
            print(
                f"Total frames: "
                f"{total_frames}"
            )

            print(
                f"Tracked frames: "
                f"{tracked_frames}"
            )

            print(
                f"Tracking rate: "
                f"{tracking_rate:.1f}%"
            )

            print(
                f"Pose detected: "
                f"{detected_frames}"
            )

            print(
                f"Pose detection rate: "
                f"{detection_rate:.1f}%"
            )

            print(
                f"Sequence shape: "
                f"{sequence.shape}"
            )

            print(
                f"Saved: "
                f"{output_file}"
            )

            print(
                f"Preview: "
                f"{preview_file}"
            )

        else:

            print(
                "\nWARNING:"
            )

            print(
                "No pose sequence was generated."
            )

            print(
                "Check the tracking preview."
            )


# ============================================================
# FINISH
# ============================================================

pose.close()

print("\n================================")
print("ALL PROCESSING FINISHED")
print("================================")
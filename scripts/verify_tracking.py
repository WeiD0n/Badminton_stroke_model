import cv2
import os

# ============================================================
# SETTINGS
# ============================================================

INPUT_FOLDER = "dataset/raw_videos/clear"
OUTPUT_FOLDER = "dataset/pose_previews/clear"


# ============================================================
# TRACKER
# ============================================================

def create_tracker():

    if hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create()

    if hasattr(cv2, "legacy") and hasattr(
        cv2.legacy,
        "TrackerCSRT_create"
    ):
        return cv2.legacy.TrackerCSRT_create()

    raise RuntimeError(
        "CSRT tracker is not available."
    )


# ============================================================
# FIND VIDEOS
# ============================================================

video_files = sorted([
    file
    for file in os.listdir(INPUT_FOLDER)
    if file.lower().endswith(
        (".mp4", ".avi", ".mov", ".mkv")
    )
])


print("================================")
print("TRACKING VERIFICATION")
print("================================")

print(
    f"Found {len(video_files)} videos."
)


# ============================================================
# PROCESS EACH VIDEO
# ============================================================

for video_index, video_file in enumerate(
    video_files,
    start=1
):

    print()
    print("================================")
    print(
        f"Video {video_index}/{len(video_files)}"
    )
    print(
        f"Processing: {video_file}"
    )
    print("================================")


    # --------------------------------------------------------
    # PATHS
    # --------------------------------------------------------

    video_path = os.path.join(
        INPUT_FOLDER,
        video_file
    )

    video_name = os.path.splitext(
        video_file
    )[0]

    output_path = os.path.join(
        OUTPUT_FOLDER,
        video_name + "_tracking.mp4"
    )


    # --------------------------------------------------------
    # OPEN VIDEO
    # --------------------------------------------------------

    cap = cv2.VideoCapture(
        video_path
    )

    if not cap.isOpened():

        print(
            "ERROR: Could not open video."
        )

        continue


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

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )


    print(
        f"Resolution: "
        f"{width} x {height}"
    )

    print(
        f"FPS: {fps:.2f}"
    )

    print(
        f"Total frames: "
        f"{total_frames}"
    )


    # --------------------------------------------------------
    # FIRST FRAME
    # --------------------------------------------------------

    success, first_frame = cap.read()

    if not success:

        print(
            "ERROR: Could not read first frame."
        )

        cap.release()
        continue


    # --------------------------------------------------------
    # SELECT TARGET PLAYER
    # --------------------------------------------------------

    print()
    print(
        "Select the TARGET PLAYER."
    )

    print(
        "Include the entire target player."
    )

    print(
        "Do NOT include another player."
    )

    print(
        "Press ENTER or SPACE."
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
        int(value)
        for value in roi
    ]


    # --------------------------------------------------------
    # CHECK ROI
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # CREATE TRACKER
    # --------------------------------------------------------

    tracker = create_tracker()

    tracker.init(
        first_frame,
        (x, y, w, h)
    )


    # --------------------------------------------------------
    # OUTPUT VIDEO
    # --------------------------------------------------------

    os.makedirs(
        OUTPUT_FOLDER,
        exist_ok=True
    )


    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )


    out = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (width, height)
    )


    # --------------------------------------------------------
    # PROCESS FRAMES
    # --------------------------------------------------------

    current_frame = first_frame

    frame_number = 0

    tracked_frames = 0


    while True:

        # ----------------------------------------------------
        # TRACK
        # ----------------------------------------------------

        if frame_number == 0:

            tracking_success = True

            bbox = (
                x,
                y,
                w,
                h
            )

        else:

            success, current_frame = cap.read()

            if not success:
                break


            tracking_success, bbox = tracker.update(
                current_frame
            )


        frame_number += 1


        # ----------------------------------------------------
        # CREATE PREVIEW
        # ----------------------------------------------------

        preview = current_frame.copy()


        # ----------------------------------------------------
        # TRACKING SUCCESS
        # ----------------------------------------------------

        if tracking_success:

            tracked_frames += 1


            tx, ty, tw, th = [
                int(value)
                for value in bbox
            ]


            # Keep inside frame

            tx = max(
                0,
                tx
            )

            ty = max(
                0,
                ty
            )


            tx2 = min(
                width,
                tx + tw
            )

            ty2 = min(
                height,
                ty + th
            )


            # ------------------------------------------------
            # GREEN TRACKING BOX
            # ------------------------------------------------

            cv2.rectangle(
                preview,
                (tx, ty),
                (tx2, ty2),
                (0, 255, 0),
                4
            )


            cv2.putText(
                preview,
                "TARGET TRACKING",
                (
                    tx,
                    max(
                        40,
                        ty - 10
                    )
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )


        # ----------------------------------------------------
        # TRACKING LOST
        # ----------------------------------------------------

        else:

            cv2.putText(
                preview,
                "TRACKING LOST",
                (40, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 0, 255),
                3
            )


        # ----------------------------------------------------
        # FRAME NUMBER
        # ----------------------------------------------------

        cv2.putText(
            preview,
            f"Frame: "
            f"{frame_number}/"
            f"{total_frames}",
            (30, height - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )


        # ----------------------------------------------------
        # WRITE PREVIEW
        # ----------------------------------------------------

        out.write(
            preview
        )


        # ----------------------------------------------------
        # PROGRESS
        # ----------------------------------------------------

        if frame_number % 10 == 0:

            print(
                f"Progress: "
                f"{frame_number}/"
                f"{total_frames} | "
                f"Tracked: "
                f"{tracked_frames}"
            )


    # --------------------------------------------------------
    # FINISH VIDEO
    # --------------------------------------------------------

    cap.release()
    out.release()


    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    tracking_rate = (
        tracked_frames
        / max(
            1,
            frame_number
        )
        * 100
    )


    print()
    print(
        f"Finished: {video_file}"
    )

    print(
        f"Tracking rate: "
        f"{tracking_rate:.1f}%"
    )

    print(
        f"Preview: "
        f"{output_path}"
    )


# ============================================================
# FINISHED
# ============================================================

print()
print("================================")
print("ALL VIDEOS VERIFIED")
print("================================")
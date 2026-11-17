
import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path

# ==================================================
# SETTINGS
# ==================================================

VIDEO_PATH = Path(
    r"C:\Users\User\Videos\FYP\Stroke_Clips\Smash\smash_001.mp4"
)

OUTPUT_PATH = Path("smash_001_comparison.mp4")
OUTPUT_IMAGE = Path("smash_001_pose_landmarks.png")

PAD_X = 0.35
PAD_Y = 0.20

# 1 = every frame
# 2 = every 2 frames
FRAME_STEP = 1

mp_pose = mp.solutions.pose
drawing = mp.solutions.drawing_utils


# ==================================================
# TRACKER
# ==================================================

def create_tracker():
    if hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create()

    if hasattr(cv2, "legacy"):
        return cv2.legacy.TrackerCSRT_create()

    raise RuntimeError(
        "CSRT tracker unavailable. Install opencv-contrib-python."
    )


def clip_box(x, y, w, h, frame_w, frame_h):
    x1 = max(0, min(frame_w - 1, int(x)))
    y1 = max(0, min(frame_h - 1, int(y)))
    x2 = max(x1 + 1, min(frame_w, int(x + w)))
    y2 = max(y1 + 1, min(frame_h, int(y + h)))

    return x1, y1, x2, y2


def get_expanded_crop(frame, box):
    frame_h, frame_w = frame.shape[:2]
    x, y, w, h = box

    pad_x = int(w * PAD_X)
    pad_y = int(h * PAD_Y)

    x1, y1, x2, y2 = clip_box(
        x - pad_x,
        y - pad_y,
        w + 2 * pad_x,
        h + 2 * pad_y,
        frame_w,
        frame_h
    )

    return frame[y1:y2, x1:x2], (x1, y1, x2, y2)


# ==================================================
# MAIN
# ==================================================

def main():

    if not VIDEO_PATH.exists():
        raise FileNotFoundError(
            f"Video not found: {VIDEO_PATH}"
        )

    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        raise RuntimeError("Could not open video.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        fps = 30.0

    success, first_frame = cap.read()

    if not success:
        raise RuntimeError("Could not read the first frame.")

    frame_h, frame_w = first_frame.shape[:2]

    # Select the badminton player manually.
    roi = cv2.selectROI(
        "Select badminton player",
        first_frame,
        showCrosshair=True,
        fromCenter=False
    )

    cv2.destroyWindow("Select badminton player")

    x, y, w, h = map(int, roi)

    if w <= 0 or h <= 0:
        cap.release()
        raise RuntimeError("No player ROI was selected.")

    tracker = create_tracker()
    tracker.init(first_frame, (x, y, w, h))

    # Ensure video dimensions are even for MP4 encoding.
    output_w = frame_w + (frame_w % 2)
    output_h = frame_h + (frame_h % 2)

    # Two vertically stacked frames.
    comparison_h = output_h * 2

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        str(OUTPUT_PATH),
        fourcc,
        fps / FRAME_STEP,
        (output_w, comparison_h)
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Could not create output video.")

    frame_index = 0
    saved_image = False
    pose_detections = 0

    print("Starting pose extraction...")
    print(f"Input frames: {total_frames}")
    print(f"Output frame interval: every {FRAME_STEP} frame(s)")

    try:

        with mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as pose:

            frame = first_frame.copy()

            while True:

                if frame_index > 0:

                    success, frame = cap.read()

                    if not success:
                        break

                    tracking_ok, tracked = tracker.update(frame)

                    if tracking_ok:
                        x, y, w, h = map(int, tracked)

                # Original frame for the top panel.
                original_frame = frame.copy()

                # Copy for skeleton overlay.
                pose_frame = frame.copy()

                crop, crop_box = get_expanded_crop(
                    frame,
                    (x, y, w, h)
                )

                pose_found = False

                if crop.size > 0:

                    rgb = cv2.cvtColor(
                        crop,
                        cv2.COLOR_BGR2RGB
                    )

                    result = pose.process(rgb)

                    if result.pose_landmarks:

                        pose_found = True
                        pose_detections += 1

                        # Draw landmarks on the expanded crop.
                        annotated_crop = crop.copy()

                        drawing.draw_landmarks(
                            annotated_crop,
                            result.pose_landmarks,
                            mp_pose.POSE_CONNECTIONS
                        )

                        # Place annotated crop back onto
                        # the corresponding original frame area.
                        x1, y1, x2, y2 = crop_box

                        pose_frame[y1:y2, x1:x2] = annotated_crop

                        # Save the first successful pose image.
                        if not saved_image:

                            cv2.imwrite(
                                str(OUTPUT_IMAGE),
                                pose_frame
                            )

                            print(
                                f"Saved image: {OUTPUT_IMAGE.resolve()}"
                            )

                            saved_image = True

                # Add labels.
                cv2.putText(
                    original_frame,
                    "ORIGINAL VIDEO",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2
                )

                label = (
                    "MEDIAPIPE POSE - 33 LANDMARKS"
                    if pose_found
                    else "MEDIAPIPE POSE - NOT DETECTED"
                )

                cv2.putText(
                    pose_frame,
                    label,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0) if pose_found else (0, 0, 255),
                    2
                )

                # Pad frames if dimensions are odd.
                if output_w != frame_w or output_h != frame_h:

                    original_frame = cv2.copyMakeBorder(
                        original_frame, 0, output_h - frame_h,
                        0, output_w - frame_w,
                        cv2.BORDER_CONSTANT,
                        value=(0, 0, 0)
                    )

                    pose_frame = cv2.copyMakeBorder(
                        pose_frame, 0, output_h - frame_h,
                        0, output_w - frame_w,
                        cv2.BORDER_CONSTANT,
                        value=(0, 0, 0)
                    )

                # Stack original on top, pose overlay below.
                comparison_frame = np.vstack([
                    original_frame,
                    pose_frame
                ])

                # Save every frame or every 2 frames.
                if frame_index % FRAME_STEP == 0:

                    writer.write(comparison_frame)

                # Live preview.
                preview = cv2.resize(
                    comparison_frame,
                    (min(1000, output_w),
                     int(comparison_h * min(1000, output_w) / output_w))
                )

                cv2.imshow(
                    "Original vs MediaPipe Pose",
                    preview
                )

                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    print("Stopped by user.")
                    break

                frame_index += 1

                if frame_index % 100 == 0:
                    print(
                        f"Processed {frame_index}/{total_frames} frames | "
                        f"Pose detections: {pose_detections}"
                    )

    finally:
        cap.release()
        writer.release()
        cv2.destroyAllWindows()

    print("\n========== COMPLETED ==========")
    print(f"Processed frames: {frame_index}")
    print(f"Successful pose detections: {pose_detections}")
    print(f"Comparison video: {OUTPUT_PATH.resolve()}")

    if saved_image:
        print(f"Pose image: {OUTPUT_IMAGE.resolve()}")
    else:
        print("No successful pose detection was saved.")


if __name__ == "__main__":
    main()


import cv2
import mediapipe as mp
import numpy as np

from pathlib import Path

from scripts.extract_pose import (
    create_tracker,
    get_expanded_crop,
    get_pose_box,
    smooth_box,
    clip_box,
    PAD_X,
    PAD_Y,
)

VIDEO_PATH = Path("dataset/raw_videos/smash/smash_592.mp4")

mp_pose = mp.solutions.pose


def main():
    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        print("Cannot open video:", VIDEO_PATH)
        return

    ok, first_frame = cap.read()

    if not ok:
        print("Cannot read first frame.")
        cap.release()
        return

    frame_h, frame_w = first_frame.shape[:2]

    selected = cv2.selectROI(
        "Select player",
        first_frame,
        showCrosshair=True,
        fromCenter=False
    )

    cv2.destroyWindow("Select player")

    x, y, w, h = map(int, selected)

    if w <= 0 or h <= 0:
        print("Invalid selection.")
        cap.release()
        return

    tracker = create_tracker()
    tracker.init(first_frame, (x, y, w, h))

    frame_index = 0

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:

        frame = first_frame

        while True:
            if frame_index > 0:
                ok, frame = cap.read()

                if not ok:
                    break

                tracking_ok, tracked = tracker.update(frame)

                if tracking_ok:
                    x, y, w, h = map(int, tracked)
                else:
                    print(f"Frame {frame_index}: TRACKER FAILURE")

            frame_h, frame_w = frame.shape[:2]

            crop, crop_box = get_expanded_crop(
                frame, (x, y, w, h)
            )

            display = frame.copy()

            if crop.size > 0:
                rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                result = pose.process(rgb)

                if result.pose_landmarks:
                    landmarks = result.pose_landmarks.landmark

                    values = np.array([
                        [lm.x, lm.y, lm.z]
                        for lm in landmarks
                    ])

                    print(
                        f"Frame {frame_index}: "
                        f"crop={crop.shape[1]}x{crop.shape[0]} | "
                        f"landmark X=[{values[:, 0].min():.3f}, "
                        f"{values[:, 0].max():.3f}] | "
                        f"box=({x},{y},{w},{h})"
                    )

                    pose_box = get_pose_box(
                        landmarks,
                        crop_box,
                        frame_w,
                        frame_h
                    )

                    if pose_box is not None:
                        updated = smooth_box(
                            (x, y, w, h),
                            pose_box
                        )

                        bx, by, bx2, by2 = clip_box(
                            updated[0],
                            updated[1],
                            updated[2],
                            updated[3],
                            frame_w,
                            frame_h
                        )

                        x, y = bx, by
                        w, h = bx2 - bx, by2 - by

                        tracker = create_tracker()
                        tracker.init(frame, (x, y, w, h))

                    mp.solutions.drawing_utils.draw_landmarks(
                        crop,
                        result.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS
                    )

                    cv2.imshow("Pose crop", crop)

            cv2.rectangle(
                display,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            cv2.putText(
                display,
                f"Frame: {frame_index}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            cv2.imshow("Original video tracking", display)

            key = cv2.waitKey(30) & 0xFF

            if key == 27:
                break

            frame_index += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

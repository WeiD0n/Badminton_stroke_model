
from pathlib import Path
import cv2
import mediapipe as mp

# ------------------------------------------------------------
# SETTINGS
# ------------------------------------------------------------

RAW_DIR = Path("dataset/raw_videos")
OUTPUT_DIR = Path("dataset/pose_reviews")

CLASSES = ["smash", "drop", "clear"]
MAX_VIDEOS_PER_CLASS = 5

PAD_X = 0.35
PAD_Y = 0.20
VISIBILITY_THRESHOLD = 0.35

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


def create_tracker():
    """Support OpenCV versions with legacy CSRT tracking."""
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
        return cv2.legacy.TrackerCSRT_create()

    if hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create()

    raise RuntimeError(
        "CSRT tracker unavailable. Try: "
        "python -m pip install opencv-contrib-python"
    )


def expand_box(box, frame_width, frame_height):
    x, y, w, h = [int(v) for v in box]

    pad_x = int(w * PAD_X)
    pad_y = int(h * PAD_Y)

    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(frame_width, x + w + pad_x)
    y2 = min(frame_height, y + h + pad_y)

    return x1, y1, x2, y2


def main():
    for class_name in CLASSES:
        class_dir = RAW_DIR / class_name

        if not class_dir.exists():
            print(f"Missing folder: {class_dir}")
            continue

        videos = sorted(class_dir.glob("*.mp4"))[
            :MAX_VIDEOS_PER_CLASS
        ]

        output_class_dir = OUTPUT_DIR / class_name
        output_class_dir.mkdir(parents=True, exist_ok=True)

        for video_path in videos:
            print(f"\nReviewing: {video_path}")

            cap = cv2.VideoCapture(str(video_path))

            if not cap.isOpened():
                print("Could not open video. Skipping.")
                continue

            ok, first_frame = cap.read()

            if not ok:
                print("Could not read first frame. Skipping.")
                cap.release()
                continue

            height, width = first_frame.shape[:2]

            # Select the intended player on the first frame.
            roi = cv2.selectROI(
                "Select intended player - ENTER to confirm",
                first_frame,
                showCrosshair=True,
                fromCenter=False
            )
            cv2.destroyWindow(
                "Select intended player - ENTER to confirm"
            )

            if roi[2] <= 0 or roi[3] <= 0:
                print("No player selected. Skipping.")
                cap.release()
                continue

            tracker = create_tracker()
            tracker.init(first_frame, tuple(map(float, roi)))

            output_path = (
                output_class_dir /
                f"{video_path.stem}_review.mp4"
            )

            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 30.0

            writer = cv2.VideoWriter(
                str(output_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (width, height)
            )

            if not writer.isOpened():
                print("Could not create output video.")
                cap.release()
                continue

            frame = first_frame
            frame_index = 0
            detected = 0
            tracking_failures = 0

            with mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            ) as pose:

                while True:
                    if frame_index > 0:
                        ok, frame = cap.read()
                        if not ok:
                            break

                    frame_index += 1

                    tracked, tracked_box = tracker.update(frame)

                    if tracked:
                        x, y, w, h = [
                            int(v) for v in tracked_box
                        ]
                    else:
                        tracking_failures += 1
                        x, y, w, h = map(int, roi)

                    x1, y1, x2, y2 = expand_box(
                        (x, y, w, h), width, height
                    )

                    crop = frame[y1:y2, x1:x2]

                    if crop.size > 0:
                        rgb = cv2.cvtColor(
                            crop, cv2.COLOR_BGR2RGB
                        )
                        result = pose.process(rgb)

                        if result.pose_landmarks:
                            detected += 1

                            # Convert crop landmarks to full-frame pixels.
                            for landmark in result.pose_landmarks.landmark:
                                if landmark.visibility < VISIBILITY_THRESHOLD:
                                    continue

                                px = int(x1 + landmark.x * (x2 - x1))
                                py = int(y1 + landmark.y * (y2 - y1))

                                cv2.circle(
                                    frame, (px, py), 3,
                                    (0, 255, 0), -1
                                )

                            # Draw skeleton in crop coordinates, then place
                            # it on the original frame.
                            overlay = frame.copy()
                            crop_overlay = overlay[y1:y2, x1:x2]

                            mp_drawing.draw_landmarks(
                                crop_overlay,
                                result.pose_landmarks,
                                mp_pose.POSE_CONNECTIONS
                            )

                            overlay[y1:y2, x1:x2] = crop_overlay
                            frame = overlay

                            # Update tracking region from detected landmarks.
                            landmarks = result.pose_landmarks.landmark
                            visible_points = [
                                (lm.x, lm.y)
                                for lm in landmarks
                                if lm.visibility >= VISIBILITY_THRESHOLD
                            ]

                            if len(visible_points) >= 8:
                                xs = [p[0] for p in visible_points]
                                ys = [p[1] for p in visible_points]

                                cx1 = int(x1 + min(xs) * (x2 - x1))
                                cy1 = int(y1 + min(ys) * (y2 - y1))
                                cx2 = int(x1 + max(xs) * (x2 - x1))
                                cy2 = int(y1 + max(ys) * (y2 - y1))

                                bw = max(20, cx2 - cx1)
                                bh = max(20, cy2 - cy1)

                                new_box = (
                                    max(0, cx1 - int(bw * PAD_X)),
                                    max(0, cy1 - int(bh * PAD_Y)),
                                    min(width - 1, bw + int(2 * bw * PAD_X)),
                                    min(height - 1, bh + int(2 * bh * PAD_Y))
                                )

                                tracker = create_tracker()
                                tracker.init(frame, tuple(map(float, new_box)))

                    # Visual indicators
                    cv2.rectangle(
                        frame, (x1, y1), (x2, y2),
                        (255, 200, 0), 2
                    )

                    cv2.putText(
                        frame,
                        f"{class_name.upper()} | Frame {frame_index} | "
                        f"Pose detections: {detected}",
                        (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (0, 255, 255), 2
                    )

                    cv2.putText(
                        frame,
                        "Press Q to stop reviewing this video",
                        (15, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (255, 255, 255), 1
                    )

                    writer.write(frame)

                    cv2.imshow("Pose Tracking Review", frame)

                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

            cap.release()
            writer.release()
            cv2.destroyAllWindows()

            rate = detected / max(frame_index, 1) * 100

            print(f"Saved preview: {output_path}")
            print(f"Frames processed: {frame_index}")
            print(f"Frames with pose: {detected} ({rate:.1f}%)")
            print(f"Tracker failures: {tracking_failures}")

    cv2.destroyAllWindows()
    print(f"\nPreviews saved under: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()

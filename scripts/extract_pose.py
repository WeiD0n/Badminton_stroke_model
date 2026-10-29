
import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
from time import perf_counter


# SETTINGS

RAW_FOLDER = Path("dataset/raw_videos")
SEQUENCE_FOLDER = Path("dataset/sequences_tracked")
VISIBILITY_FOLDER = Path("dataset/visibility_tracked")

CLASSES = ["smash", "drop", "clear"]
TARGET_FRAMES = 30
TARGET_TOTAL_PER_CLASS = 200
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

# Expand the tracker box before sending the crop to MediaPipe.
PAD_X = 0.35
PAD_Y = 0.20

# Build a padded box around detected pose landmarks.
POSE_BOX_PADDING_X = 0.35
POSE_BOX_PADDING_Y = 0.25
LANDMARK_VISIBILITY_THRESHOLD = 0.35
MIN_VISIBLE_LANDMARKS = 8

# Smooth pose-derived box updates.
POSE_BOX_SMOOTHING = 0.65

# Flag videos below this detection rate.
MIN_DETECTION_RATE = 0.80

mp_pose = mp.solutions.pose
drawing = mp.solutions.drawing_utils


# ============================================================
# TRACKER
# ============================================================
def create_tracker():
    if hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create()

    if hasattr(cv2, "legacy") and hasattr(
        cv2.legacy, "TrackerCSRT_create"
    ):
        return cv2.legacy.TrackerCSRT_create()

    raise RuntimeError(
        "CSRT tracker unavailable. Install opencv-contrib-python."
    )


def clip_box(x, y, w, h, frame_w, frame_h):
    """Clip an x, y, width, height box to the frame."""
    x1 = max(0, min(frame_w - 1, int(x)))
    y1 = max(0, min(frame_h - 1, int(y)))
    x2 = max(x1 + 1, min(frame_w, int(x + w)))
    y2 = max(y1 + 1, min(frame_h, int(y + h)))

    return x1, y1, x2, y2


def get_expanded_crop(frame, box):
    """
    Return the expanded crop and its exact coordinates in
    the original frame.
    """
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


def get_pose_box(landmarks, crop_box, frame_width, frame_height):
    """
    Convert visible MediaPipe landmarks from crop coordinates
    into a padded bounding box in original-frame coordinates.
    """
    crop_x1, crop_y1, crop_x2, crop_y2 = crop_box
    crop_width = crop_x2 - crop_x1
    crop_height = crop_y2 - crop_y1

    if crop_width <= 0 or crop_height <= 0:
        return None

    points = []

    for landmark in landmarks:
        if landmark.visibility >= LANDMARK_VISIBILITY_THRESHOLD:
            px = crop_x1 + landmark.x * crop_width
            py = crop_y1 + landmark.y * crop_height
            points.append((px, py))

    if len(points) < MIN_VISIBLE_LANDMARKS:
        return None

    xs = [point[0] for point in points]
    ys = [point[1] for point in points]

    left = min(xs)
    right = max(xs)
    top = min(ys)
    bottom = max(ys)

    body_width = max(right - left, 1)
    body_height = max(bottom - top, 1)

    pad_x = body_width * POSE_BOX_PADDING_X
    pad_y = body_height * POSE_BOX_PADDING_Y

    left = max(0, int(left - pad_x))
    top = max(0, int(top - pad_y))
    right = min(frame_width, int(right + pad_x))
    bottom = min(frame_height, int(bottom + pad_y))

    if right <= left or bottom <= top:
        return None

    return left, top, right - left, bottom - top


def smooth_box(old_box, new_box, alpha=POSE_BOX_SMOOTHING):
    """Smooth box movement to reduce sudden changes."""
    return tuple(
        int(alpha * new_value + (1 - alpha) * old_value)
        for old_value, new_value in zip(old_box, new_box)
    )


# ============================================================
# RESAMPLE TO 30 FRAMES
# ============================================================
def resample_sequence(sequence, target_frames=TARGET_FRAMES):
    sequence = np.asarray(sequence, dtype=np.float32)

    if len(sequence) == 0:
        return np.zeros(
            (target_frames, sequence.shape[1]),
            dtype=np.float32
        )

    if len(sequence) == 1:
        return np.repeat(sequence, target_frames, axis=0)

    old_positions = np.linspace(0, len(sequence) - 1, len(sequence))
    new_positions = np.linspace(0, len(sequence) - 1, target_frames)

    output = np.empty(
        (target_frames, sequence.shape[1]),
        dtype=np.float32
    )

    for column in range(sequence.shape[1]):
        output[:, column] = np.interp(
            new_positions,
            old_positions,
            sequence[:, column]
        )

    return output


# ============================================================
# DISCOVER VIDEOS
# ============================================================

def discover_jobs():
    jobs = []

    for class_name in CLASSES:
        input_folder = RAW_FOLDER / class_name
        sequence_folder = SEQUENCE_FOLDER / class_name
        visibility_folder = VISIBILITY_FOLDER / class_name

        if not input_folder.exists():
            print(f"WARNING: Missing folder: {input_folder}")
            continue

        sequence_folder.mkdir(parents=True, exist_ok=True)
        visibility_folder.mkdir(parents=True, exist_ok=True)

        # Count existing valid sequence/visibility pairs.
        valid_existing = 0

        for seq_path in sequence_folder.glob("*.npy"):
            vis_path = (
                visibility_folder /
                f"{seq_path.stem}_visibility.npy"
            )

            if not vis_path.exists():
                print(f"WARNING: Missing visibility file: {vis_path}")
                continue

            try:
                seq = np.load(seq_path, allow_pickle=False)
                vis = np.load(vis_path, allow_pickle=False)

                if (
                    seq.shape == (TARGET_FRAMES, 99)
                    and vis.shape == (TARGET_FRAMES, 33)
                    and np.isfinite(seq).all()
                    and np.isfinite(vis).all()
                    and vis.min() >= 0
                    and vis.max() <= 1
                ):
                    valid_existing += 1
                else:
                    print(
                        f"WARNING: Invalid existing pair: "
                        f"{class_name}/{seq_path.stem}"
                    )

            except Exception as exc:
                print(
                    f"WARNING: Cannot validate {seq_path.name}: {exc}"
                )

        remaining = TARGET_TOTAL_PER_CLASS - valid_existing

        print(
            f"\n{class_name.upper()}: "
            f"{valid_existing}/{TARGET_TOTAL_PER_CLASS} valid sequences; "
            f"need {max(0, remaining)} more."
        )

        if remaining <= 0:
            print("Target already reached; skipping this class.")
            continue

        videos = sorted(
            [
                path for path in input_folder.iterdir()
                if path.is_file()
                and path.suffix.lower() in VIDEO_EXTENSIONS
            ],
            key=lambda path: path.name.lower()
        )

        added = 0

        for video_path in videos:
            sequence_path = (
                sequence_folder / f"{video_path.stem}.npy"
            )
            visibility_path = (
                visibility_folder /
                f"{video_path.stem}_visibility.npy"
            )

            # Never overwrite existing outputs.
            if sequence_path.exists() or visibility_path.exists():
                continue

            jobs.append((
                class_name,
                video_path,
                sequence_path,
                visibility_path
            ))

            added += 1

            if added >= remaining:
                break

        if added < remaining:
            print(
                f"WARNING: Only {added} new videos available for "
                f"{class_name}; target may not be reached."
            )

    return jobs



# ============================================================
# PROCESS ONE VIDEO
# ============================================================
def extract_video(video_path, sequence_path, visibility_path):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"ERROR: Cannot open {video_path.name}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, first_frame = cap.read()

    if not ok:
        cap.release()
        print(f"ERROR: Cannot read {video_path.name}")
        return None

    frame_h, frame_w = first_frame.shape[:2]

    # Manually select the intended player once.
    window_name = f"Select player - {video_path.name}"

    print(
        "\nSelect the intended player in the first frame."
        "\nDraw a box around the entire visible body."
        "\nInclude head, arms, torso and legs where possible."
        "\nPress ENTER or SPACE to confirm; C to cancel."
    )

    selected = cv2.selectROI(
        window_name,
        first_frame,
        showCrosshair=True,
        fromCenter=False
    )

    cv2.destroyWindow(window_name)

    x, y, w, h = map(int, selected)

    if w <= 0 or h <= 0:
        print(f"Skipped: {video_path.name}")
        cap.release()
        return None

    tracker = create_tracker()
    tracker.init(first_frame, (x, y, w, h))

    coordinates = []
    visibilities = []

    frame_index = 0
    tracker_failures = 0
    pose_detections = 0
    stop_video = False
    stop_all = False

    try:
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
                        # Keep the last known box if tracking fails.
                        tracker_failures += 1

                frame_h, frame_w = frame.shape[:2]

                # Track box -> expanded crop -> pose landmarks.
                crop, crop_box = get_expanded_crop(
                    frame, (x, y, w, h)
                )

                features = np.zeros(99, dtype=np.float32)
                visibility = np.zeros(33, dtype=np.float32)
                pose_found = False
                crop_preview = crop.copy()

                if crop.size > 0:
                    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    result = pose.process(rgb)

                    if result.pose_landmarks:
                        pose_found = True
                        pose_detections += 1

                        landmarks = result.pose_landmarks.landmark

                        # Keep the same 99-feature representation:
                        # 33 landmarks x (x, y, z), relative to hips.
                        left_hip = landmarks[23]
                        right_hip = landmarks[24]

                        hip_x = (left_hip.x + right_hip.x) / 2.0
                        hip_y = (left_hip.y + right_hip.y) / 2.0
                        hip_z = (left_hip.z + right_hip.z) / 2.0

                        values = []

                        for landmark in landmarks:
                            values.extend([
                                landmark.x - hip_x,
                                landmark.y - hip_y,
                                landmark.z - hip_z
                            ])

                        features = np.asarray(
                            values, dtype=np.float32
                        )

                        visibility = np.asarray(
                            [
                                float(landmark.visibility)
                                for landmark in landmarks
                            ],
                            dtype=np.float32
                        )

                        drawing.draw_landmarks(
                            crop_preview,
                            result.pose_landmarks,
                            mp_pose.POSE_CONNECTIONS
                        )

                        # Update the tracking box using the detected
                        # player's visible body landmarks.
                        pose_box = get_pose_box(
                            landmarks,
                            crop_box,
                            frame_w,
                            frame_h
                        )

                        if pose_box is not None:
                            proposed_box = pose_box

                            # Smooth the new box against the current one.
                            old_box = (x, y, w, h)
                            updated_box = smooth_box(
                                old_box, proposed_box
                            )

                            bx, by, bx2, by2 = clip_box(
                                updated_box[0],
                                updated_box[1],
                                updated_box[2],
                                updated_box[3],
                                frame_w,
                                frame_h
                            )

                            x, y = bx, by
                            w, h = bx2 - bx, by2 - by

                            # Reinitialize CSRT on the updated body box,
                            # so the next frame follows the new position.
                            tracker = create_tracker()
                            tracker.init(frame, (x, y, w, h))

                coordinates.append(features)
                visibilities.append(visibility)

                # Preview.
                display = frame.copy()

                color = (
                    (0, 220, 0) if pose_found
                    else (0, 0, 255)
                )

                cv2.rectangle(
                    display,
                    (x, y),
                    (x + w, y + h),
                    color,
                    3
                )

                detection_rate = (
                    pose_detections / (frame_index + 1)
                ) * 100

                cv2.putText(
                    display,
                    f"{video_path.name} | "
                    f"Frame {frame_index + 1}/{total_frames}",
                    (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2
                )

                cv2.putText(
                    display,
                    f"Pose: {'YES' if pose_found else 'NO'} | "
                    f"Detection: {detection_rate:.1f}%",
                    (10, 56),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    color,
                    2
                )

                cv2.putText(
                    display,
                    "Q = skip video | ESC = stop all",
                    (10, 84),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2
                )

                cv2.imshow("Landmark-guided player tracking", display)

                if crop_preview.size > 0:
                    cv2.imshow("Expanded crop and pose", crop_preview)

                key = cv2.waitKey(1) & 0xFF
                frame_index += 1

                if key == ord("q"):
                    stop_video = True
                    break

                if key == 27:
                    stop_all = True
                    break

                if frame_index % 100 == 0:
                    print(
                        f"  {video_path.name}: "
                        f"{frame_index}/{total_frames} frames | "
                        f"poses={pose_detections}"
                    )

    finally:
        cap.release()
        cv2.destroyAllWindows()

    # Never save an incomplete video.
    if stop_video or stop_all:
        print(
            f"WARNING: {video_path.name} was interrupted; "
            "no output saved."
        )
        return {"stop_all": stop_all, "result": None}

    if not coordinates:
        print(f"WARNING: No frames extracted from {video_path.name}")
        return {"stop_all": False, "result": None}

    coordinates = np.asarray(coordinates, dtype=np.float32)
    visibilities = np.asarray(visibilities, dtype=np.float32)

    detection_rate = pose_detections / len(coordinates)

    # Resample both arrays to 30 frames.
    coordinates_30 = resample_sequence(coordinates)
    visibility_30 = resample_sequence(visibilities)

    sequence_path.parent.mkdir(parents=True, exist_ok=True)
    visibility_path.parent.mkdir(parents=True, exist_ok=True)

    np.save(sequence_path, coordinates_30)
    np.save(visibility_path, visibility_30)

    result = {
        "video": video_path.name,
        "frames": len(coordinates),
        "pose_detections": pose_detections,
        "detection_rate": detection_rate,
        "tracker_failures": tracker_failures,
        "low_detection": detection_rate < MIN_DETECTION_RATE
    }

    print(
        f"\nRESULT: {video_path.name}"
        f"\n  Original frames: {len(coordinates)}"
        f"\n  Pose detections: {pose_detections}/{len(coordinates)}"
        f"\n  Detection rate: {detection_rate * 100:.1f}%"
        f"\n  Tracker failures: {tracker_failures}"
        f"\n  Saved sequence: {coordinates_30.shape}"
        f"\n  Saved visibility: {visibility_30.shape}"
    )

    if result["low_detection"]:
        print("  WARNING: Low pose detection rate; review this video.")

    return {"stop_all": False, "result": result}


# ============================================================
# MAIN
# ============================================================
def main():
    jobs = discover_jobs()

    if not jobs:
        print("No videos found under dataset/raw_videos/")
        return

    print("=" * 60)
    print(f"VIDEOS TO PROCESS: {len(jobs)}")
    print(f"CLASSES: {', '.join(CLASSES)}")
    print(f"TARGET FRAMES: {TARGET_FRAMES}")
    print(f"SEQUENCES OUTPUT: {SEQUENCE_FOLDER}")
    print(f"VISIBILITY OUTPUT: {VISIBILITY_FOLDER}")
    print("=" * 60)

    print(
        "\nIMPORTANT:"
        "\n- Each video needs one initial player selection."
        "\n- Landmark-guided boxes update as the player moves."
        "\n- Results go to separate folders."
        "\n- Review low-detection videos before training."
    )

    input("\nPress ENTER to start...")

    start = perf_counter()
    results = []
    skipped = 0

    for index, job in enumerate(jobs, 1):
        class_name, video_path, sequence_path, visibility_path = job

        print(
            f"\n[{index}/{len(jobs)}] "
            f"{class_name.upper()}: {video_path.name}"
        )

        try:
            outcome = extract_video(
                video_path,
                sequence_path,
                visibility_path
            )

            if outcome is None:
                skipped += 1
                continue

            if outcome["result"] is not None:
                results.append({
                    "class": class_name,
                    **outcome["result"]
                })
            else:
                skipped += 1

            if outcome["stop_all"]:
                print("Extraction stopped by user.")
                break

        except Exception as exc:
            skipped += 1
            print(f"ERROR processing {video_path.name}: {exc}")

    elapsed = (perf_counter() - start) / 60

    print("\n" + "=" * 60)
    print("EXTRACTION SUMMARY")
    print("=" * 60)
    print(f"Completed videos: {len(results)}")
    print(f"Skipped/failed videos: {skipped}")
    print(f"Elapsed time: {elapsed:.1f} minutes")

    for class_name in CLASSES:
        class_results = [
            item for item in results
            if item["class"] == class_name
        ]

        low = sum(item["low_detection"] for item in class_results)

        print(
            f"{class_name.upper()}: "
            f"{len(class_results)} completed, "
            f"{low} low-detection videos"
        )

    print(f"\nSequences: {SEQUENCE_FOLDER}")
    print(f"Visibility: {VISIBILITY_FOLDER}")
    print("Inspect outputs before training.")


if __name__ == "__main__":
    main()

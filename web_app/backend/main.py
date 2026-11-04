
from pathlib import Path
import tempfile
import sys

import cv2
import mediapipe as mp
import numpy as np

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware


# ============================================================
# FASTAPI CONFIGURATION
# ============================================================

app = FastAPI(title="Rally Badminton Analytics API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# AUTOENCODER CONFIGURATION
# ============================================================

# Expected location:
# <project-root>/backend/app/main.py

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.evaluate_form import evaluate_form


CLASSES = ["Clear", "Drop Shot", "Smash"]

STROKE_CLASS_MAP = {
    "Clear": "clear",
    "Drop Shot": "drop",
    "Smash": "smash"
}

ALLOWED_EXTENSIONS = {
    ".mp4", ".mov", ".webm", ".avi", ".mkv"
}

MAX_FILE_SIZE = 200 * 1024 * 1024
TARGET_FRAMES = 30


# Keep these values consistent with the training extraction script.

PAD_X = 0.35
PAD_Y = 0.20

POSE_BOX_PADDING_X = 0.35
POSE_BOX_PADDING_Y = 0.25

LANDMARK_VISIBILITY_THRESHOLD = 0.35
MIN_VISIBLE_LANDMARKS = 8

POSE_BOX_SMOOTHING = 0.65
MIN_DETECTION_RATE = 0.80


mp_pose = mp.solutions.pose


# ============================================================
# TRACKING AND CROP HELPERS
# ============================================================

def create_tracker():
    """Create a CSRT tracker across supported OpenCV versions."""

    if hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create()

    if hasattr(cv2, "legacy") and hasattr(
        cv2.legacy, "TrackerCSRT_create"
    ):
        return cv2.legacy.TrackerCSRT_create()

    raise RuntimeError(
        "CSRT tracker is unavailable. Install opencv-contrib-python "
        "in the same Python environment as the backend."
    )


def clip_box(x, y, w, h, frame_w, frame_h):
    """Clip an x, y, width, height box to image boundaries."""

    x1 = max(0, min(frame_w - 1, int(x)))
    y1 = max(0, min(frame_h - 1, int(y)))

    x2 = max(x1 + 1, min(frame_w, int(x + w)))
    y2 = max(y1 + 1, min(frame_h, int(y + h)))

    return x1, y1, x2, y2


def get_expanded_crop(frame, box):
    """Expand the tracker box and return crop and original bounds."""

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
        frame_h,
    )

    return frame[y1:y2, x1:x2], (x1, y1, x2, y2)


def get_pose_box(
    landmarks,
    crop_box,
    frame_width,
    frame_height
):
    """Build a padded full-frame box from visible pose landmarks."""

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

    left, right = min(xs), max(xs)
    top, bottom = min(ys), max(ys)

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


def smooth_box(
    old_box,
    new_box,
    alpha=POSE_BOX_SMOOTHING
):
    """Smooth tracker box updates."""

    return tuple(
        int(alpha * new_value + (1 - alpha) * old_value)
        for old_value, new_value in zip(old_box, new_box)
    )


# ============================================================
# SEQUENCE RESAMPLING
# ============================================================

def resample_sequence(
    sequence,
    target_frames=TARGET_FRAMES
):
    sequence = np.asarray(sequence, dtype=np.float32)

    if sequence.ndim != 2:
        raise ValueError(
            f"Expected a 2D sequence (frames, features), "
            f"got {sequence.shape}"
        )

    if len(sequence) == 0:
        raise ValueError("No video frames were processed.")

    if len(sequence) == 1:
        return np.repeat(
            sequence,
            target_frames,
            axis=0
        )

    old_positions = np.linspace(
        0,
        len(sequence) - 1,
        len(sequence)
    )

    new_positions = np.linspace(
        0,
        len(sequence) - 1,
        target_frames
    )

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
# MEDIAPIPE EXTRACTION WITH PLAYER SELECTION AND CSRT TRACKING
# ============================================================

def extract_pose_sequence(video_path):
    """
    Match the training extraction pipeline:

    1. Manual first-frame player selection.
    2. CSRT tracking.
    3. Expanded player crops.
    4. Hip-relative 99-feature coordinates.
    5. Landmark-guided tracking updates.
    6. Resampling to 30 frames.

    Requires a desktop session because cv2.selectROI opens a window.
    """

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError("Could not open this video.")

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    success, first_frame = cap.read()

    if not success:
        cap.release()
        raise ValueError(
            "Could not read the first frame of this video."
        )

    frame_h, frame_w = first_frame.shape[:2]

    window_name = (
        "Select player - draw box around entire visible body"
    )

    print(
        "\nSelect the intended player in the first frame. "
        "Draw a box around the visible body, then press "
        "ENTER/SPACE to confirm. Press C to cancel."
    )

    try:

        selected = cv2.selectROI(
            window_name,
            first_frame,
            showCrosshair=True,
            fromCenter=False
        )

    except cv2.error as exc:

        cap.release()
        cv2.destroyAllWindows()

        raise RuntimeError(
            "Player selection window could not open. "
            "Run the backend in a desktop session that "
            "supports OpenCV windows."
        ) from exc

    finally:
        cv2.destroyWindow(window_name)

    x, y, w, h = map(int, selected)

    if w <= 0 or h <= 0:
        cap.release()

        raise ValueError(
            "Player selection was cancelled or the "
            "selection box was empty."
        )

    tracker = create_tracker()
    tracker.init(first_frame, (x, y, w, h))

    coordinates = []
    visibilities = []

    frame_index = 0
    pose_detections = 0
    tracker_failures = 0

    try:

        with mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as pose:

            frame = first_frame

            while True:

                if frame_index > 0:

                    success, frame = cap.read()

                    if not success:
                        break

                    tracking_ok, tracked = tracker.update(frame)

                    if tracking_ok:

                        x, y, w, h = map(
                            int,
                            tracked
                        )

                    else:

                        # Retain the last known box.
                        tracker_failures += 1

                frame_h, frame_w = frame.shape[:2]

                crop, crop_box = get_expanded_crop(
                    frame,
                    (x, y, w, h)
                )

                features = np.zeros(
                    99,
                    dtype=np.float32
                )

                visibility = np.zeros(
                    33,
                    dtype=np.float32
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

                        landmarks = (
                            result.pose_landmarks.landmark
                        )

                        left_hip = landmarks[23]
                        right_hip = landmarks[24]

                        hip_x = (
                            left_hip.x + right_hip.x
                        ) / 2.0

                        hip_y = (
                            left_hip.y + right_hip.y
                        ) / 2.0

                        hip_z = (
                            left_hip.z + right_hip.z
                        ) / 2.0

                        values = []

                        for landmark in landmarks:

                            values.extend([
                                landmark.x - hip_x,
                                landmark.y - hip_y,
                                landmark.z - hip_z,
                            ])

                        features = np.asarray(
                            values,
                            dtype=np.float32
                        )

                        visibility = np.asarray(
                            [
                                float(landmark.visibility)
                                for landmark in landmarks
                            ],
                            dtype=np.float32
                        )

                        pose_box = get_pose_box(
                            landmarks,
                            crop_box,
                            frame_w,
                            frame_h
                        )

                        if pose_box is not None:

                            proposed_box = pose_box

                            updated_box = smooth_box(
                                (x, y, w, h),
                                proposed_box
                            )

                            bx, by, bx2, by2 = clip_box(
                                updated_box[0],
                                updated_box[1],
                                updated_box[2],
                                updated_box[3],
                                frame_w,
                                frame_h,
                            )

                            x, y = bx, by
                            w, h = bx2 - bx, by2 - by

                            # Reinitialize CSRT around updated body box.
                            tracker = create_tracker()

                            tracker.init(
                                frame,
                                (x, y, w, h)
                            )

                coordinates.append(features)
                visibilities.append(visibility)

                # Display tracking progress.
                display = frame.copy()

                box_color = (
                    (0, 220, 0)
                    if pose_found
                    else (0, 0, 255)
                )

                cv2.rectangle(
                    display,
                    (x, y),
                    (x + w, y + h),
                    box_color,
                    2
                )

                detection_rate = (
                    pose_detections /
                    (frame_index + 1) * 100.0
                )

                cv2.putText(
                    display,
                    f"Frame {frame_index + 1}/{total_frames} | "
                    f"Pose: {'YES' if pose_found else 'NO'} | "
                    f"Detection: {detection_rate:.1f}%",
                    (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    box_color,
                    2,
                )

                cv2.putText(
                    display,
                    "Q = cancel video",
                    (10, 56),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

                cv2.imshow(
                    "Tracked player",
                    display
                )

                key = cv2.waitKey(1) & 0xFF

                frame_index += 1

                if key == ord("q") or key == 27:

                    raise ValueError(
                        "Video analysis cancelled during tracking."
                    )

    finally:

        cap.release()
        cv2.destroyAllWindows()

    if not coordinates:
        raise ValueError(
            "No frames were extracted from this video."
        )

    detection_rate = (
        pose_detections / len(coordinates)
    )

    if pose_detections == 0:

        raise ValueError(
            "MediaPipe could not detect a pose inside "
            "the selected player crop. Try selecting "
            "the whole visible body more tightly."
        )

    sequence = resample_sequence(
        np.asarray(coordinates, dtype=np.float32)
    )

    visibility_sequence = resample_sequence(
        np.asarray(visibilities, dtype=np.float32)
    )

    return sequence, {
        "frame_count": len(coordinates),
        "reported_video_frames": total_frames,
        "pose_detections": pose_detections,
        "pose_detection_rate": round(
            detection_rate,
            4
        ),
        "low_detection_warning": (
            detection_rate < MIN_DETECTION_RATE
        ),
        "tracker_failures": tracker_failures,
        "selected_player_tracking": True,
        "expanded_crops": True,
        "landmark_guided_tracking": True,
        "visibility_mean": round(
            float(np.mean(visibility_sequence)),
            4
        ),
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/api/health")
def health():

    return {
        "status": "ok",
        "evaluation_method": "LSTM Autoencoder",
        "autoencoder_evaluation_enabled": True,
        "reference_profiles_enabled": True,
        "input_shape": [TARGET_FRAMES, 99],
        "full_feature_shape": [TARGET_FRAMES, 139],
        "classes": CLASSES,
        "player_selection_enabled": True,
        "csrt_tracking_enabled": True,
        "expanded_crops_enabled": True,
        "landmark_guided_tracking_enabled": True
    }


# ============================================================
# VIDEO ANALYSIS AND LSTM AUTOENCODER FORM EVALUATION
# ============================================================

@app.post("/api/analyse")
async def analyse_video(
    file: UploadFile = File(...),
    stroke_type: str = Form(...),
):

    extension = Path(
        file.filename or ""
    ).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:

        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported video format. "
                "Use MP4, MOV, WEBM, AVI, or MKV."
            )
        )

    if stroke_type not in CLASSES:

        raise HTTPException(
            status_code=400,
            detail="Select Clear, Drop Shot, or Smash."
        )

    contents = await file.read(
        MAX_FILE_SIZE + 1
    )

    if not contents:

        raise HTTPException(
            status_code=400,
            detail="The video is empty."
        )

    if len(contents) > MAX_FILE_SIZE:

        raise HTTPException(
            status_code=413,
            detail="Video must be smaller than 200 MB."
        )

    temp_path = None

    try:

        # ----------------------------------------------------
        # 1. SAVE TEMPORARY VIDEO
        # ----------------------------------------------------

        with tempfile.NamedTemporaryFile(
            suffix=extension,
            delete=False
        ) as temp_file:

            temp_file.write(contents)
            temp_path = Path(temp_file.name)

        # ----------------------------------------------------
        # 2. EXTRACT POSE SEQUENCE
        # ----------------------------------------------------

        sequence, video_info = extract_pose_sequence(
            temp_path
        )

        if sequence.shape != (
            TARGET_FRAMES,
            99
        ):

            raise ValueError(
                f"Expected pose sequence shape "
                f"({TARGET_FRAMES}, 99), "
                f"got {sequence.shape}"
            )

        # ----------------------------------------------------
        # 3. SELECT MATCHING AUTOENCODER
        # ----------------------------------------------------

        autoencoder_class = STROKE_CLASS_MAP[
            stroke_type
        ]

        # ----------------------------------------------------
        # 4. RUN OBJECTIVE 3 FORM EVALUATION
        # ----------------------------------------------------

        form_analysis = evaluate_form(
            pose_sequence=sequence,
            predicted_class=autoencoder_class
        )

        # ----------------------------------------------------
        # 5. TRACKING WARNING
        # ----------------------------------------------------

        warning = None

        if video_info["low_detection_warning"]:

            warning = (
                "Pose detection was below 80%. "
                "Review the player selection and tracking. "
                "The form evaluation may be unreliable."
            )

        # ----------------------------------------------------
        # 6. RETURN RESULT
        # ----------------------------------------------------

        return {
            "status": "success",

            "evaluation_method": "LSTM Autoencoder",

            "filename": file.filename,

            "selected_stroke_type": stroke_type,

            "video_info": video_info,

            "form_analysis": form_analysis,

            "warning": warning,

            "message": (
                "Form evaluation completed using "
                "class-specific professional reference "
                "profiles, biomechanical feature comparison, "
                "and LSTM autoencoder reconstruction."
            )
        }

    except ValueError as exc:

        raise HTTPException(
            status_code=422,
            detail=str(exc)
        ) from exc

    except (OSError, RuntimeError) as exc:

        raise HTTPException(
            status_code=500,
            detail=f"Form evaluation failed: {exc}"
        ) from exc

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=f"Video analysis failed: {exc}"
        ) from exc

    finally:

        if temp_path is not None:
            temp_path.unlink(missing_ok=True)

        await file.close()

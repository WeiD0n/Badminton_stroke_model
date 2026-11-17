
from pathlib import Path
from datetime import datetime, timedelta, timezone
import tempfile
import sys
import os
import uuid
import mimetypes
import re
import json

import boto3

import cv2
import mediapipe as mp
import numpy as np

from fastapi import (
    Body,
    Response,
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
    Depends,
    status,
    Query,
)

from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from .database import get_db
from .models import User, Analysis
from .schemas import UserRegister, UserLogin, UserResponse, TokenResponse

from .auth import (
    hash_password,
    verify_password,
    create_access_token,
    get_current_user,
    get_optional_current_user,
    AUTH_COOKIE_NAME,
    ACCESS_TOKEN_EXPIRE_MINUTES
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from starlette.background import BackgroundTask


# ============================================================
# FASTAPI CONFIGURATION
# ============================================================

app = FastAPI(title="Rally Badminton Analytics API")


FRONTEND_ORIGIN = os.getenv(
    "FRONTEND_ORIGIN",
    "http://localhost:5173"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in FRONTEND_ORIGIN.split(",") if origin.strip()],
    allow_credentials=True,
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
from .stroke_classifier import predict_stroke


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


# ============================================================
# CLOUDFLARE R2 CONFIGURATION
# ============================================================

VIDEO_RETENTION_DAYS = int(os.getenv("VIDEO_RETENTION_DAYS", "90"))


def get_r2_client():
    """Create an S3-compatible Cloudflare R2 client from environment settings."""
    account_id = os.getenv("R2_ACCOUNT_ID")
    access_key_id = os.getenv("R2_ACCESS_KEY_ID")
    secret_access_key = os.getenv("R2_SECRET_ACCESS_KEY")
    endpoint_url = os.getenv("R2_ENDPOINT_URL") or (
        f"https://{account_id}.r2.cloudflarestorage.com" if account_id else None
    )

    if not all([account_id, access_key_id, secret_access_key, endpoint_url]):
        raise RuntimeError("Cloudflare R2 credentials or endpoint are missing from .env.")

    return boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key,
        region_name="auto",
    )


def get_r2_bucket_name():
    bucket_name = os.getenv("R2_BUCKET_NAME")
    if not bucket_name:
        raise RuntimeError("R2_BUCKET_NAME is missing from .env.")
    return bucket_name


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

def extract_pose_sequence(video_path, roi):
    """
    Match the training extraction pipeline:

    1. Use the manually selected first-frame ROI supplied by the browser.
    2. CSRT tracking.
    3. Expanded player crops.
    4. Hip-relative 99-feature coordinates.
    5. Landmark-guided tracking updates.
    6. Resampling to 30 frames.

    The browser supplies normalised ROI coordinates; no desktop GUI is used.
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

    # Convert browser-normalised ROI coordinates to source-video pixels.
    roi_x, roi_y, roi_width, roi_height = roi
    if not all(np.isfinite(value) for value in roi):
        cap.release()
        raise ValueError("Player selection contains invalid coordinates.")
    if (
        roi_x < 0 or roi_y < 0 or roi_width <= 0 or roi_height <= 0
        or roi_x >= 1 or roi_y >= 1
        or roi_x + roi_width > 1.000001
        or roi_y + roi_height > 1.000001
    ):
        cap.release()
        raise ValueError("Player selection is outside the video frame or has an invalid size.")

    x = int(round(roi_x * frame_w))
    y = int(round(roi_y * frame_h))
    w = int(round(roi_width * frame_w))
    h = int(round(roi_height * frame_h))

    x1, y1, x2, y2 = clip_box(x, y, w, h, frame_w, frame_h)
    x, y, w, h = x1, y1, x2 - x1, y2 - y1

    if w < 2 or h < 2:
        cap.release()
        raise ValueError("Player selection is too small. Draw a larger box around the player.")

    tracker = create_tracker()
    tracker_initialised = tracker.init(first_frame, (x, y, w, h))
    if tracker_initialised is False:
        cap.release()
        raise RuntimeError("CSRT could not initialise with the selected player box.")

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

                frame_index += 1

    finally:

        cap.release()

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
        "selection_method": "browser_manual_roi",
        "expanded_crops": True,
        "landmark_guided_tracking": True,
        "visibility_mean": round(
            float(np.mean(visibility_sequence)),
            4
        ),
    }




# ============================================================
# SKELETON OVERLAY VIDEO GENERATION
# ============================================================

def draw_pose_skeleton(frame, landmarks, crop_box):
    """Draw visible MediaPipe pose landmarks and connections on the full frame."""
    crop_x1, crop_y1, crop_x2, crop_y2 = crop_box
    crop_width = crop_x2 - crop_x1
    crop_height = crop_y2 - crop_y1

    if crop_width <= 0 or crop_height <= 0:
        return frame

    points = {}
    for index, landmark in enumerate(landmarks):
        if landmark.visibility >= LANDMARK_VISIBILITY_THRESHOLD:
            px = int(crop_x1 + landmark.x * crop_width)
            py = int(crop_y1 + landmark.y * crop_height)
            points[index] = (px, py)

    # Draw bones first so the joint markers remain visible.
    for start_index, end_index in mp_pose.POSE_CONNECTIONS:
        if start_index in points and end_index in points:
            cv2.line(
                frame,
                points[start_index],
                points[end_index],
                (80, 220, 120),
                3,
                cv2.LINE_AA,
            )

    for point in points.values():
        cv2.circle(frame, point, 5, (30, 180, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, point, 7, (255, 255, 255), 1, cv2.LINE_AA)

    return frame


def cleanup_overlay_file(path):
    """Remove a generated overlay after FastAPI has finished sending it."""
    Path(path).unlink(missing_ok=True)


def generate_skeleton_overlay_video(video_path, roi, output_path):
    """
    Generate a playable video with a MediaPipe skeleton overlay.

    Tracking and crop updates intentionally mirror extract_pose_sequence()
    so the overlay follows the same manually selected player.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError("Could not open this video.")

    success, first_frame = cap.read()
    if not success:
        cap.release()
        raise ValueError("Could not read the first frame of this video.")

    frame_h, frame_w = first_frame.shape[:2]
    roi_x, roi_y, roi_width, roi_height = roi

    if not all(np.isfinite(value) for value in roi):
        cap.release()
        raise ValueError("Player selection contains invalid coordinates.")

    if (
        roi_x < 0 or roi_y < 0 or roi_width <= 0 or roi_height <= 0
        or roi_x >= 1 or roi_y >= 1
        or roi_x + roi_width > 1.000001
        or roi_y + roi_height > 1.000001
    ):
        cap.release()
        raise ValueError(
            "Player selection is outside the video frame or has an invalid size."
        )

    x = int(round(roi_x * frame_w))
    y = int(round(roi_y * frame_h))
    w = int(round(roi_width * frame_w))
    h = int(round(roi_height * frame_h))
    x1, y1, x2, y2 = clip_box(x, y, w, h, frame_w, frame_h)
    x, y, w, h = x1, y1, x2 - x1, y2 - y1

    if w < 2 or h < 2:
        cap.release()
        raise ValueError(
            "Player selection is too small. Draw a larger box around the player."
        )

    tracker = create_tracker()
    tracker_initialised = tracker.init(first_frame, (x, y, w, h))
    if tracker_initialised is False:
        cap.release()
        raise RuntimeError("CSRT could not initialise with the selected player box.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps <= 0 or fps > 120:
        fps = 30.0

    # VP8/WebM is broadly supported by modern browsers. Fall back to MP4V
    # if this OpenCV/FFmpeg build cannot create a WebM writer.
    output_width = frame_w + (frame_w % 2)
    output_height = frame_h + (frame_h % 2)
    writer = None
    output_extension = Path(output_path).suffix.lower()

    if output_extension == ".webm":
        for codec in ("VP80", "VP90"):
            candidate = cv2.VideoWriter(
                str(output_path),
                cv2.VideoWriter_fourcc(*codec),
                fps,
                (output_width, output_height),
            )
            if candidate.isOpened():
                writer = candidate
                break
            candidate.release()

    if writer is None:
        output_path = Path(output_path).with_suffix(".mp4")
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (output_width, output_height),
        )

    if not writer.isOpened():
        cap.release()
        writer.release()
        raise RuntimeError(
            "Could not create the annotated video. Check that OpenCV has FFmpeg video codec support."
        )

    frame_index = 0
    pose_detections = 0

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
                        x, y, w, h = map(int, tracked)

                frame_h, frame_w = frame.shape[:2]
                crop, crop_box = get_expanded_crop(frame, (x, y, w, h))
                annotated_frame = frame.copy()

                if crop.size > 0:
                    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    result = pose.process(rgb)

                    if result.pose_landmarks:
                        pose_detections += 1
                        landmarks = result.pose_landmarks.landmark

                        annotated_frame = draw_pose_skeleton(
                            annotated_frame,
                            landmarks,
                            crop_box,
                        )

                        pose_box = get_pose_box(
                            landmarks,
                            crop_box,
                            frame_w,
                            frame_h,
                        )

                        if pose_box is not None:
                            updated_box = smooth_box(
                                (x, y, w, h),
                                pose_box,
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

                            tracker = create_tracker()
                            tracker.init(frame, (x, y, w, h))

                # Some WebM encoders require even frame dimensions.
                if (
                    annotated_frame.shape[1] != output_width
                    or annotated_frame.shape[0] != output_height
                ):
                    annotated_frame = cv2.copyMakeBorder(
                        annotated_frame,
                        0,
                        output_height - annotated_frame.shape[0],
                        0,
                        output_width - annotated_frame.shape[1],
                        cv2.BORDER_CONSTANT,
                        value=(0, 0, 0),
                    )

                writer.write(annotated_frame)
                frame_index += 1

    finally:
        cap.release()
        writer.release()

    if frame_index == 0:
        Path(output_path).unlink(missing_ok=True)
        raise ValueError("No frames were processed for the skeleton overlay.")

    if pose_detections == 0:
        Path(output_path).unlink(missing_ok=True)
        raise ValueError(
            "MediaPipe could not detect the selected player. Try selecting the whole visible body more tightly."
        )

    return Path(output_path), {
        "frame_count": frame_index,
        "pose_detections": pose_detections,
        "pose_detection_rate": round(pose_detections / frame_index, 4),
        "codec": "webm" if Path(output_path).suffix.lower() == ".webm" else "mp4",
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/api/health")
def health():

    return {
        "status": "ok",
        "evaluation_method": "LSTM Classifier + LSTM Autoencoder",
        "stroke_classification_enabled": True,
        "classification_model": "run_059",
        "classification_input_shape": [TARGET_FRAMES, 139],
        "autoencoder_evaluation_enabled": True,
        "reference_profiles_enabled": True,
        "input_shape": [TARGET_FRAMES, 99],
        "full_feature_shape": [TARGET_FRAMES, 139],
        "classes": CLASSES,
        "player_selection_enabled": True,
        "player_selection_method": "browser_manual_roi",
        "csrt_tracking_enabled": True,
        "expanded_crops_enabled": True,
        "landmark_guided_tracking_enabled": True,
        "skeleton_overlay_enabled": True
    }


# ============================================================
# VIDEO ANALYSIS AND LSTM AUTOENCODER FORM EVALUATION
# ============================================================


@app.post("/api/analyse")
async def analyse_video(
    file: UploadFile = File(...),
    stroke_type: str = Form(...),
    save_analysis: bool = Form(False),
    save_video: bool = Form(False),
    roi_x: float = Form(...),
    roi_y: float = Form(...),
    roi_width: float = Form(...),
    roi_height: float = Form(...),
    db: Session = Depends(get_db),
    current_user: User | None = Depends(get_optional_current_user),
):
    """Analyse a video; optionally persist results and the source video."""
    if save_analysis and current_user is None:
        raise HTTPException(status_code=401, detail="Log in to save analysis results.")

    if save_video and not save_analysis:
        raise HTTPException(
            status_code=400,
            detail="Saving the video also requires saving its analysis."
        )

    extension = Path(file.filename or "").suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Unsupported video format. Use MP4, MOV, WEBM, AVI, or MKV."
        )

    if stroke_type not in CLASSES:
        raise HTTPException(status_code=400, detail="Select Clear, Drop Shot, or Smash.")

    contents = await file.read(MAX_FILE_SIZE + 1)
    if not contents:
        raise HTTPException(status_code=400, detail="The video is empty.")
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="Video must be smaller than 200 MB.")

    temp_path = None
    uploaded_object_key = None

    try:
        # Preserve the existing temporary-file, pose extraction and LSTM workflow.
        with tempfile.NamedTemporaryFile(suffix=extension, delete=False) as temp_file:
            temp_file.write(contents)
            temp_path = Path(temp_file.name)

        selected_roi = (roi_x, roi_y, roi_width, roi_height)
        sequence, video_info = extract_pose_sequence(temp_path, selected_roi)
        if sequence.shape != (TARGET_FRAMES, 99):
            raise ValueError(
                f"Expected pose sequence shape ({TARGET_FRAMES}, 99), got {sequence.shape}"
            )

        # Objective 2: predict the stroke independently with the run-059 LSTM classifier.
        # Its preprocessing uses the same 30x99 pose sequence, then derives
        # 12 biomechanical + 28 temporal features and applies run-059 statistics.
        classification = predict_stroke(sequence, PROJECT_ROOT)

        # Objective 3: retain the user's selected stroke as the reference class
        # for the existing autoencoder form-evaluation workflow.
        autoencoder_class = STROKE_CLASS_MAP[stroke_type]
        form_analysis = evaluate_form(
            pose_sequence=sequence,
            predicted_class=autoencoder_class
        )

        warning = None
        if video_info["low_detection_warning"]:
            warning = (
                "Pose detection was below 80%. Review the player selection and tracking. "
                "The form evaluation may be unreliable."
            )

        response_data = {
            "status": "success",
            "evaluation_method": "LSTM Autoencoder",
            "filename": file.filename,
            "selected_stroke_type": stroke_type,
            "classification": classification,
            "classification_matches_selected_stroke": (
                classification["predicted_class"] == STROKE_CLASS_MAP[stroke_type]
            ),
            "video_info": video_info,
            "form_analysis": form_analysis,
            "warning": warning,
            "message": (
                "Form evaluation completed using class-specific professional reference "
                "profiles, biomechanical feature comparison, and LSTM autoencoder reconstruction."
            ),
        }

        saved_analysis_id = None
        video_expires_at = None

        # Save only when an authenticated user explicitly opts in.
        if save_analysis and current_user is not None:
            safe_filename = Path(file.filename or "video").name
            object_key = None

            if save_video:
                object_key = (
                    f"users/{current_user.id}/analyses/{uuid.uuid4().hex}{extension}"
                )
                video_expires_at = datetime.now(timezone.utc) + timedelta(
                    days=VIDEO_RETENTION_DAYS
                )

                try:
                    get_r2_client().put_object(
                        Bucket=get_r2_bucket_name(),
                        Key=object_key,
                        Body=contents,
                        ContentType=file.content_type or "application/octet-stream",
                    )
                    uploaded_object_key = object_key
                except Exception as exc:
                    raise HTTPException(
                        status_code=502,
                        detail=f"Could not save the video to cloud storage: {exc}"
                    ) from exc

            form_values = form_analysis if isinstance(form_analysis, dict) else {}
            analysis_record = Analysis(
                user_id=current_user.id,
                stroke_type=stroke_type,
                form_score=form_values.get("form_score"),
                biomechanical_score=form_values.get("biomechanical_score"),
                temporal_similarity_score=form_values.get("temporal_similarity_score"),
                reconstruction_error=form_values.get("reconstruction_error"),
                result_data=jsonable_encoder(response_data),
                video_saved=bool(save_video),
                video_object_key=object_key,
                video_filename=safe_filename if save_video else None,
                video_expires_at=video_expires_at,
            )

            try:
                db.add(analysis_record)
                db.commit()
                db.refresh(analysis_record)
                saved_analysis_id = str(analysis_record.id)
            except Exception:
                db.rollback()
                if uploaded_object_key:
                    try:
                        get_r2_client().delete_object(
                            Bucket=get_r2_bucket_name(),
                            Key=uploaded_object_key,
                        )
                    except Exception:
                        pass
                raise

        response_data["saved"] = bool(saved_analysis_id)
        response_data["analysis_id"] = saved_analysis_id
        response_data["video_saved"] = bool(save_video and saved_analysis_id)
        response_data["video_expires_at"] = (
            video_expires_at.isoformat() if video_expires_at and saved_analysis_id else None
        )
        return response_data

    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (OSError, RuntimeError) as exc:
        raise HTTPException(status_code=500, detail=f"Form evaluation failed: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Video analysis failed: {exc}") from exc
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        await file.close()




# ============================================================
# SKELETON OVERLAY API
# ============================================================

@app.post("/api/skeleton-overlay")
async def create_skeleton_overlay(
    file: UploadFile = File(...),
    roi_x: float = Form(...),
    roi_y: float = Form(...),
    roi_width: float = Form(...),
    roi_height: float = Form(...),
):
    """
    Generate and return an annotated copy of the uploaded video.
    The original upload and existing analysis records are not modified.
    """
    extension = Path(file.filename or "").suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Unsupported video format. Use MP4, MOV, WEBM, AVI, or MKV.",
        )

    contents = await file.read(MAX_FILE_SIZE + 1)
    if not contents:
        raise HTTPException(status_code=400, detail="The video is empty.")
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Video must be smaller than 200 MB.",
        )

    input_path = None
    output_path = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=extension,
            delete=False,
        ) as temp_file:
            temp_file.write(contents)
            input_path = Path(temp_file.name)

        with tempfile.NamedTemporaryFile(
            suffix=".webm",
            delete=False,
        ) as output_file:
            output_path = Path(output_file.name)

        output_path.unlink(missing_ok=True)

        output_path, overlay_info = generate_skeleton_overlay_video(
            input_path,
            (roi_x, roi_y, roi_width, roi_height),
            output_path,
        )

        media_type = (
            "video/webm"
            if output_path.suffix.lower() == ".webm"
            else "video/mp4"
        )
        download_name = (
            "skeleton_overlay.webm"
            if media_type == "video/webm"
            else "skeleton_overlay.mp4"
        )

        return FileResponse(
            path=str(output_path),
            media_type=media_type,
            filename=download_name,
            background=BackgroundTask(cleanup_overlay_file, output_path),
            headers={
                "X-Pose-Detections": str(overlay_info["pose_detections"]),
                "X-Processed-Frames": str(overlay_info["frame_count"]),
            },
        )

    except HTTPException:
        if output_path is not None:
            output_path.unlink(missing_ok=True)
        raise
    except ValueError as exc:
        if output_path is not None:
            output_path.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (OSError, RuntimeError) as exc:
        if output_path is not None:
            output_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=500,
            detail=f"Skeleton overlay generation failed: {exc}",
        ) from exc
    except Exception as exc:
        if output_path is not None:
            output_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=500,
            detail=f"Could not generate skeleton overlay: {exc}",
        ) from exc
    finally:
        if input_path is not None:
            input_path.unlink(missing_ok=True)
        await file.close()


# ============================================================
# ANALYSIS HISTORY
# ============================================================

@app.get("/api/analyses")
def get_analysis_history(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
):
    """Return only the signed-in user's saved analyses, newest first."""
    records = (
        db.query(Analysis)
        .filter(Analysis.user_id == current_user.id)
        .order_by(Analysis.created_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )

    return {
        "items": [
            {
                "id": str(record.id),
                "stroke_type": record.stroke_type,
                "form_score": record.form_score,
                "biomechanical_score": record.biomechanical_score,
                "temporal_similarity_score": record.temporal_similarity_score,
                "reconstruction_error": record.reconstruction_error,
                "video_saved": record.video_saved,
                "video_filename": record.video_filename,
                "video_expires_at": record.video_expires_at,
                "created_at": record.created_at,
                "result": record.result_data,
            }
            for record in records
        ],
        "count": len(records),
        "limit": limit,
        "offset": offset,
    }


@app.get("/api/analyses/{analysis_id}")
def get_analysis_detail(
    analysis_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Return one saved analysis only if it belongs to the current user."""
    record = (
        db.query(Analysis)
        .filter(
            Analysis.id == analysis_id,
            Analysis.user_id == current_user.id,
        )
        .first()
    )

    if record is None:
        raise HTTPException(status_code=404, detail="Analysis not found.")

    return {
        "id": str(record.id),
        "stroke_type": record.stroke_type,
        "video_saved": record.video_saved,
        "video_filename": record.video_filename,
        "video_expires_at": record.video_expires_at,
        "created_at": record.created_at,
        "result": record.result_data,
    }




# ============================================================
# DOWNLOADABLE ANALYSIS REPORT
# ============================================================

@app.post("/api/analyses/report")
def download_analysis_report(
    payload: dict = Body(...),
    current_user: User = Depends(get_optional_current_user),
    db: Session = Depends(get_db),
):
    """
    Generate a PDF report for either a saved analysis (analysis_id) or
    the current unsaved result (analysis_data). Reports are generated on
    demand and are not stored in the database.
    """
    from io import BytesIO
    from xml.sax.saxutils import escape
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        KeepTogether, HRFlowable
    )

    analysis_id = payload.get("analysis_id")
    if analysis_id:
        if current_user is None:
            raise HTTPException(status_code=401, detail="Please log in to download a saved analysis report.")
        try:
            parsed_id = uuid.UUID(str(analysis_id))
        except (ValueError, TypeError, AttributeError):
            raise HTTPException(status_code=400, detail="Invalid analysis ID.")
        record = (
            db.query(Analysis)
            .filter(
                Analysis.id == parsed_id,
                Analysis.user_id == current_user.id,
            )
            .first()
        )
        if record is None:
            raise HTTPException(status_code=404, detail="Analysis not found.")
        data = record.result_data or {}
        created_at = record.created_at
        filename = record.video_filename or data.get("filename") or "Badminton video"
        stroke_type = record.stroke_type or data.get("selected_stroke_type") or "Not specified"
    else:
        data = payload.get("analysis_data")
        if not isinstance(data, dict):
            raise HTTPException(status_code=400, detail="Analysis data is required.")
        if len(jsonable_encoder(data).__str__()) > 2_000_000:
            raise HTTPException(status_code=413, detail="Analysis report data is too large.")
        created_at = datetime.now(timezone.utc)
        filename = data.get("filename") or "Badminton video"
        stroke_type = data.get("selected_stroke_type") or "Not specified"

    form = data.get("form_analysis") or {}
    video = data.get("video_info") or {}
    detail = form.get("detailed_feedback") or {}

    def safe(value):
        if value is None:
            return "Not available"
        if isinstance(value, (dict, list)):
            value = json.dumps(value, ensure_ascii=False, default=str)
        return escape(str(value))

    def number(value, digits=2, suffix=""):
        try:
            return f"{float(value):.{digits}f}{suffix}"
        except (TypeError, ValueError):
            return "Not available"

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4, rightMargin=18*mm, leftMargin=18*mm,
        topMargin=17*mm, bottomMargin=17*mm,
        title="Rally Badminton Analytics - Analysis Report",
        author="Rally Badminton Analytics",
    )
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="ReportTitle", parent=styles["Title"], fontName="Helvetica-Bold",
        fontSize=20, leading=25, textColor=colors.HexColor("#14213d"),
        alignment=TA_CENTER, spaceAfter=5,
    ))
    styles.add(ParagraphStyle(
        name="ReportSub", parent=styles["Normal"], fontSize=9,
        textColor=colors.HexColor("#64748b"), alignment=TA_CENTER, spaceAfter=14,
    ))
    styles.add(ParagraphStyle(
        name="SectionHead", parent=styles["Heading2"], fontSize=12,
        leading=15, textColor=colors.HexColor("#14213d"),
        spaceBefore=12, spaceAfter=7,
    ))
    styles.add(ParagraphStyle(
        name="SmallBody", parent=styles["BodyText"], fontSize=8.5,
        leading=12, spaceAfter=4,
    ))
    styles.add(ParagraphStyle(
        name="Cell", parent=styles["BodyText"], fontSize=8,
        leading=10, spaceAfter=0,
    ))

    story = [
        Paragraph("RALLY BADMINTON ANALYTICS", styles["ReportTitle"]),
        Paragraph("BADMINTON STROKE ANALYSIS REPORT", styles["ReportSub"]),
        HRFlowable(width="100%", thickness=1, color=colors.HexColor("#dbe3ee")),
        Spacer(1, 8),
        Paragraph("Analysis information", styles["SectionHead"]),
    ]
    created_text = created_at.strftime("%d %B %Y, %H:%M UTC") if hasattr(created_at, "strftime") else str(created_at)
    metadata = [
        ["Video filename", safe(filename)],
        ["Analysis date", safe(created_text)],
        ["Selected stroke", safe(stroke_type)],
        ["Evaluation method", safe(data.get("evaluation_method", "LSTM Autoencoder"))],
    ]
    table = Table(metadata, colWidths=[42*mm, 130*mm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (0,-1), colors.HexColor("#f1f5f9")),
        ("TEXTCOLOR", (0,0), (-1,-1), colors.HexColor("#243247")),
        ("FONTNAME", (0,0), (0,-1), "Helvetica-Bold"),
        ("FONTNAME", (1,0), (-1,-1), "Helvetica"),
        ("FONTSIZE", (0,0), (-1,-1), 8),
        ("GRID", (0,0), (-1,-1), .35, colors.HexColor("#dbe3ee")),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("LEFTPADDING", (0,0), (-1,-1), 7),
        ("RIGHTPADDING", (0,0), (-1,-1), 7),
        ("TOPPADDING", (0,0), (-1,-1), 6),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
    ]))
    story.append(table)

    story.append(Paragraph("Stroke form evaluation", styles["SectionHead"]))
    metrics = [
        ["Overall form score", number(form.get("form_score")) + " / 100"],
        ["Form rating", safe(form.get("form_rating"))],
        ["Biomechanical score", number(form.get("biomechanical_score")) + " / 100"],
        ["Temporal similarity", number(form.get("temporal_similarity_score")) + " / 100"],
        ["Reconstruction error", number(form.get("reconstruction_error"), 6)],
    ]
    metric_table = Table(metrics, colWidths=[75*mm, 97*mm])
    metric_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (0,-1), colors.HexColor("#f1f5f9")),
        ("GRID", (0,0), (-1,-1), .35, colors.HexColor("#dbe3ee")),
        ("FONTNAME", (0,0), (0,-1), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,-1), 8.5),
        ("LEFTPADDING", (0,0), (-1,-1), 7),
        ("TOPPADDING", (0,0), (-1,-1), 6),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
    ]))
    story.append(metric_table)

    story.append(Paragraph("Video processing information", styles["SectionHead"]))
    video_rows = [
        ["Frames processed", safe(video.get("frame_count"))],
        ["Pose detection rate", number(float(video.get("pose_detection_rate", 0))*100, 1, "%") if video.get("pose_detection_rate") is not None else "Not available"],
        ["Average landmark visibility", number(float(video.get("visibility_mean", 0))*100, 1, "%") if video.get("visibility_mean") is not None else "Not available"],
        ["Tracker failures", safe(video.get("tracker_failures"))],
    ]
    video_table = Table(video_rows, colWidths=[75*mm, 97*mm])
    video_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (0,-1), colors.HexColor("#f1f5f9")),
        ("GRID", (0,0), (-1,-1), .35, colors.HexColor("#dbe3ee")),
        ("FONTNAME", (0,0), (0,-1), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,-1), 8.5),
        ("LEFTPADDING", (0,0), (-1,-1), 7),
        ("TOPPADDING", (0,0), (-1,-1), 6),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
    ]))
    story.append(video_table)

    if data.get("warning"):
        story.extend([
            Paragraph("Analysis notice", styles["SectionHead"]),
            Paragraph(safe(data.get("warning")), styles["SmallBody"]),
        ])

    for section_key, section_title in [
        ("interpretation", "Evaluation interpretation"),
        ("score_interpretation", "Score interpretation"),
    ]:
        value = detail.get(section_key) or form.get(section_key)
        if value:
            story.extend([
                Paragraph(section_title, styles["SectionHead"]),
                Paragraph(safe(value), styles["SmallBody"]),
            ])

    for list_key, title in [
        ("strengths", "Identified strengths"),
        ("areas_to_review", "Areas to review"),
    ]:
        items = detail.get(list_key)
        if isinstance(items, list) and items:
            story.append(Paragraph(title, styles["SectionHead"]))
            for item in items:
                if isinstance(item, dict):
                    label = item.get("feature") or item.get("name") or item.get("title") or "Observation"
                    desc = item.get("interpretation") or item.get("description") or item.get("message")
                    deviation = item.get("deviation")
                    line = f"<b>{safe(label)}</b>"
                    if desc:
                        line += f" — {safe(desc)}"
                    if deviation is not None:
                        line += f" (deviation: {safe(deviation)})"
                else:
                    line = safe(item)
                story.append(Paragraph("• " + line, styles["SmallBody"]))

    measurements = detail.get("all_measurements")
    if not isinstance(measurements, list):
        measurements = []
        for key in ("joint_angles", "torso_alignment", "elbow_angle_changes",
                    "angular_velocity", "angular_acceleration", "movement_timing"):
            values = detail.get(key)
            if isinstance(values, list):
                measurements.extend(values)
    if measurements:
        story.append(Paragraph("Detailed movement measurements", styles["SectionHead"]))
        measurement_rows = [["Measurement", "Observed value", "Reference / notes"]]
        for item in measurements[:60]:
            if isinstance(item, dict):
                measurement_rows.append([
                    Paragraph(safe(item.get("feature") or item.get("name") or item.get("metric") or "Measurement"), styles["Cell"]),
                    Paragraph(safe(item.get("value") if item.get("value") is not None else item.get("observed_value")), styles["Cell"]),
                    Paragraph(safe(item.get("interpretation") or item.get("reference") or item.get("description")), styles["Cell"]),
                ])
            else:
                measurement_rows.append([Paragraph(safe(item), styles["Cell"]), "", ""])
        mt = Table(measurement_rows, colWidths=[48*mm, 34*mm, 90*mm], repeatRows=1)
        mt.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#e8eef6")),
            ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
            ("GRID", (0,0), (-1,-1), .3, colors.HexColor("#dbe3ee")),
            ("VALIGN", (0,0), (-1,-1), "TOP"),
            ("FONTSIZE", (0,0), (-1,-1), 7.5),
            ("LEFTPADDING", (0,0), (-1,-1), 5),
            ("RIGHTPADDING", (0,0), (-1,-1), 5),
            ("TOPPADDING", (0,0), (-1,-1), 5),
            ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ]))
        story.append(mt)

    story.extend([
        Spacer(1, 14),
        HRFlowable(width="100%", thickness=.5, color=colors.HexColor("#dbe3ee")),
        Spacer(1, 5),
        Paragraph(
            "This report presents the results returned by Rally Badminton Analytics. "
            "The selected stroke is the reference category chosen for this analysis; "
            "it is not a separately reported stroke-classification confidence score.",
            styles["SmallBody"],
        ),
    ])

    doc.build(story)
    buffer.seek(0)
    safe_stroke = re.sub(r"[^A-Za-z0-9_-]+", "_", str(stroke_type)).strip("_") or "analysis"
    report_name = f"rally_{safe_stroke}_report.pdf"
    return StreamingResponse(
        buffer,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{report_name}"'},
    )



@app.get("/api/analyses/{analysis_id}/video")
def get_analysis_video(
    analysis_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Return a short-lived, signed Cloudflare R2 URL for a saved video.
    The analysis must belong to the currently authenticated user.
    """
    record = (
        db.query(Analysis)
        .filter(
            Analysis.id == analysis_id,
            Analysis.user_id == current_user.id,
        )
        .first()
    )

    # Do not reveal whether an analysis exists for another user.
    if record is None:
        raise HTTPException(status_code=404, detail="Analysis not found.")

    if not record.video_saved or not record.video_object_key:
        raise HTTPException(
            status_code=404,
            detail="No saved video is available for this analysis.",
        )

    if record.video_expires_at is not None:
        expires_at = record.video_expires_at
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
        if expires_at <= datetime.now(timezone.utc):
            raise HTTPException(
                status_code=410,
                detail="This saved video has expired.",
            )

    filename = Path(record.video_filename or "badminton-video").name
    content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"

    try:
        client = get_r2_client()
        # Ensure the object still exists before returning a playback link.
        client.head_object(
            Bucket=get_r2_bucket_name(),
            Key=record.video_object_key,
        )
        signed_url = client.generate_presigned_url(
            "get_object",
            Params={
                "Bucket": get_r2_bucket_name(),
                "Key": record.video_object_key,
                "ResponseContentDisposition": f'inline; filename="{filename}"',
                "ResponseContentType": content_type,
            },
            ExpiresIn=900,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail="Could not retrieve the saved video from cloud storage.",
        ) from exc

    return {
        "video_url": signed_url,
        "filename": filename,
        "expires_in": 900,
    }


# ============================================================
# USER AUTHENTICATION
# ============================================================

@app.post(
    "/api/auth/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED
)
def register_user(
    user_data: UserRegister,
    db: Session = Depends(get_db)
):
    # Normalise email and username
    email = str(user_data.email).strip().lower()
    username = user_data.username.strip()

    if not username:
        raise HTTPException(
            status_code=400,
            detail="Username cannot be empty."
        )

    # Check whether email already exists
    existing_user = db.query(User).filter(
        User.email == email
    ).first()

    if existing_user:
        raise HTTPException(
            status_code=409,
            detail="An account with this email already exists."
        )

    # Create user with securely hashed password
    new_user = User(
        email=email,
        username=username,
        password_hash=hash_password(user_data.password)
    )

    try:
        db.add(new_user)
        db.commit()
        db.refresh(new_user)

    except IntegrityError:
        db.rollback()

        raise HTTPException(
            status_code=409,
            detail="An account with this email already exists."
        )

    return new_user


@app.post("/api/auth/login")
def login_user(
    login_data: UserLogin,
    response: Response,
    db: Session = Depends(get_db)
):
    email = str(login_data.email).strip().lower()
    user = db.query(User).filter(User.email == email).first()

    if not user or not verify_password(login_data.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Incorrect email or password.")
    if not user.is_active:
        raise HTTPException(status_code=403, detail="This account is inactive.")

    access_token = create_access_token(str(user.id))
    cookie_secure = os.getenv("AUTH_COOKIE_SECURE", "false").lower() == "true"
    cookie_samesite = os.getenv("AUTH_COOKIE_SAMESITE", "lax").lower()
    if cookie_samesite not in {"lax", "strict", "none"}:
        cookie_samesite = "lax"

    response.set_cookie(
        key=AUTH_COOKIE_NAME,
        value=access_token,
        max_age=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        httponly=True,
        secure=cookie_secure,
        samesite=cookie_samesite,
        path="/",
    )
    return {"message": "Login successful."}


@app.post("/api/auth/logout")
def logout_user(response: Response):
    response.delete_cookie(
        key=AUTH_COOKIE_NAME,
        path="/",
        httponly=True,
        secure=os.getenv("AUTH_COOKIE_SECURE", "false").lower() == "true",
        samesite=os.getenv("AUTH_COOKIE_SAMESITE", "lax").lower(),
    )
    return {"message": "Logged out successfully."}


@app.get(
    "/api/auth/me",
    response_model=UserResponse
)
def get_my_account(
    current_user: User = Depends(get_current_user)
):
    return current_user


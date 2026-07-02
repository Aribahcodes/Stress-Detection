"""
src/physio_branch.py — Physiological Branch (80% weight in fusion)
===================================================================
ARCHITECTURE (from system diagram — Physio Branch):

  Webcam Frame (BGR, 640×480)
        ↓
  MediaPipe FaceMesh (478 landmarks + iris refinement)
        ↓
  6 Physiological Signals extracted:
    1. Blink Rate      ×0.25  (most important)
    2. Eye Openness    ×0.16
    3. Gaze Jitter     ×0.16
    4. Head Motion     ×0.12
    5. Brow Tension    ×0.08
    6. Jaw Tension     ×0.08
        ↓
  Physio Stress Score (0.0 – 1.0, weight = 0.80)

MediaPipe landmark indices used:
  Eyes  : LEFT_EYE [33,160,158,133,153,144], RIGHT_EYE [362,385,387,263,373,380]
  Iris  : LEFT_IRIS [468,469,470,471,472],   RIGHT_IRIS [473,474,475,476,477]
  Brows : LEFT_BROW [70,63,105,66,107],      RIGHT_BROW [336,296,334,293,300]
  Jaw   : [152, 148, 176, 149, 150, 136, 172, 58, 132]
  Nose  : [1] (for head motion reference)

COMPATIBILITY NOTE:
  MediaPipe 0.10.18+ removed mp.solutions in favour of mp.tasks.
  This module supports BOTH APIs with automatic fallback:
    - New API (>=0.10.18): mediapipe.tasks.python.vision.FaceLandmarker
    - Legacy API (<0.10.18): mediapipe.solutions.face_mesh
  Pin to mediapipe>=0.10.0,<0.10.18 if you prefer the legacy API,
  or keep latest and this module will use the new Tasks API.
"""

import cv2
import numpy as np
from collections import deque
from typing import Optional

# ── MediaPipe import — supports both old and new API ──────────────────────────
_MP_AVAILABLE   = False
_USE_TASKS_API  = False   # True = new mp.tasks API; False = legacy mp.solutions

try:
    import mediapipe as mp

    # Detect which API is available
    if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
        # Legacy API (mediapipe < 0.10.18)
        _mp_face_mesh  = mp.solutions.face_mesh
        _mp_drawing    = mp.solutions.drawing_utils
        _MP_AVAILABLE  = True
        _USE_TASKS_API = False
        print("[PhysioBranch] Using legacy MediaPipe solutions API.")

    elif hasattr(mp, "tasks"):
        # New Tasks API (mediapipe >= 0.10.18)
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
        import urllib.request, os, tempfile

        _MP_AVAILABLE  = True
        _USE_TASKS_API = True
        print("[PhysioBranch] Using new MediaPipe Tasks API (>= 0.10.18).")

    else:
        print("[PhysioBranch] mediapipe installed but no supported API found.")

except ImportError:
    print("[PhysioBranch] mediapipe not installed. Run: pip install mediapipe")
    print("   Physio branch will return zero scores (CNN-only mode).")


# ── MediaPipe landmark indices ────────────────────────────────────────────────
LEFT_EYE_IDX   = [33,  160, 158, 133, 153, 144]
RIGHT_EYE_IDX  = [362, 385, 387, 263, 373, 380]
LEFT_IRIS_IDX  = [468, 469, 470, 471, 472]
RIGHT_IRIS_IDX = [473, 474, 475, 476, 477]
LEFT_BROW_IDX  = [70,  63,  105, 66,  107]
RIGHT_BROW_IDX = [336, 296, 334, 293, 300]
JAW_IDX        = [152, 148, 176, 149, 150, 136, 172, 58, 132]
NOSE_TIP_IDX   = 1

PHYSIO_SIGNAL_WEIGHTS = {
    "blink_rate":    0.25,
    "eye_openness":  0.16,
    "gaze_jitter":   0.16,
    "head_motion":   0.12,
    "brow_tension":  0.08,
    "jaw_tension":   0.08,
}
TOTAL_WEIGHT = sum(PHYSIO_SIGNAL_WEIGHTS.values())  # 0.85


# ── Model file path for Tasks API ─────────────────────────────────────────────
_TASKS_MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
_TASKS_MODEL_PATH = None   # set lazily on first use


def _get_tasks_model_path() -> Optional[str]:
    """
    Download the FaceLandmarker .task model file if not already cached.
    Stores in a persistent temp directory so it survives session restarts.
    Returns local path or None on failure.
    """
    global _TASKS_MODEL_PATH
    if _TASKS_MODEL_PATH and os.path.exists(_TASKS_MODEL_PATH):
        return _TASKS_MODEL_PATH

    cache_dir = os.path.join(tempfile.gettempdir(), "mediapipe_models")
    os.makedirs(cache_dir, exist_ok=True)
    local_path = os.path.join(cache_dir, "face_landmarker.task")

    if not os.path.exists(local_path):
        print(f"[PhysioBranch] Downloading FaceLandmarker model (~28 MB)…")
        try:
            urllib.request.urlretrieve(_TASKS_MODEL_URL, local_path)
            print(f"[PhysioBranch] Model saved to {local_path}")
        except Exception as e:
            print(f"[PhysioBranch] Model download failed: {e}")
            return None

    _TASKS_MODEL_PATH = local_path
    return local_path


# ── Helper functions (shared by both APIs) ────────────────────────────────────

def _lm_to_arr(landmark, img_w: int, img_h: int) -> np.ndarray:
    """Convert a single landmark to pixel-space numpy array."""
    return np.array([landmark.x * img_w, landmark.y * img_h])


def _eye_aspect_ratio(landmarks, eye_idx: list, img_w: int, img_h: int) -> float:
    """
    EAR = (||p2-p6|| + ||p3-p5||) / (2 × ||p1-p4||).
    EAR < 0.2 → blink.
    """
    pts = [_lm_to_arr(landmarks[i], img_w, img_h) for i in eye_idx]
    p1, p2, p3, p4, p5, p6 = pts
    return float(
        (np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5))
        / (2.0 * np.linalg.norm(p1 - p4) + 1e-8)
    )


def _iris_center(landmarks, iris_idx: list, img_w: int, img_h: int) -> np.ndarray:
    pts = np.array([_lm_to_arr(landmarks[i], img_w, img_h) for i in iris_idx])
    return pts.mean(axis=0)


def _brow_distance(landmarks, brow_idx, eye_idx, img_w, img_h) -> float:
    brow_pts = np.array([_lm_to_arr(landmarks[i], img_w, img_h) for i in brow_idx])
    eye_pts  = np.array([_lm_to_arr(landmarks[i], img_w, img_h) for i in eye_idx])
    face_h   = abs(landmarks[10].y - landmarks[152].y) * img_h + 1e-8
    return float(abs(brow_pts[:, 1].mean() - eye_pts[:, 1].mean()) / face_h)


def _jaw_openness(landmarks, jaw_idx, img_w, img_h) -> float:
    pts    = np.array([_lm_to_arr(landmarks[i], img_w, img_h) for i in jaw_idx])
    face_h = abs(landmarks[10].y - landmarks[152].y) * img_h + 1e-8
    return float((pts[:, 1].max() - pts[:, 1].min()) / face_h)


def _compute_signals_from_landmarks(
    landmarks,
    img_w: int,
    img_h: int,
    blink_history: deque,
    gaze_history: deque,
    nose_history: deque,
    eye_closed_ref: list,   # [bool] mutable single-element list
    baseline_brow_ref: list,
    baseline_jaw_ref: list,
    frame_count: int,
) -> tuple[float, dict]:
    """
    Core signal extraction — shared between legacy and Tasks API paths.
    landmark list must support integer indexing with .x / .y / .z attributes.
    """
    # ── Signal 1: Blink Rate ─────────────────────────────────────────────────
    left_ear  = _eye_aspect_ratio(landmarks, LEFT_EYE_IDX,  img_w, img_h)
    right_ear = _eye_aspect_ratio(landmarks, RIGHT_EYE_IDX, img_w, img_h)
    avg_ear   = (left_ear + right_ear) / 2.0

    is_closed   = avg_ear < 0.20
    blink_event = int(is_closed and not eye_closed_ref[0])
    eye_closed_ref[0] = is_closed
    blink_history.append(blink_event)

    bps         = sum(blink_history) / (len(blink_history) / 30.0 + 1e-8)
    blink_score = float(np.clip((bps - 0.1) / 0.5, 0.0, 1.0))

    # ── Signal 2: Eye Openness ───────────────────────────────────────────────
    eye_open_score = float(np.clip(1.0 - (avg_ear / 0.30), 0.0, 1.0))

    # ── Signal 3: Gaze Jitter ────────────────────────────────────────────────
    left_iris  = _iris_center(landmarks, LEFT_IRIS_IDX,  img_w, img_h)
    right_iris = _iris_center(landmarks, RIGHT_IRIS_IDX, img_w, img_h)
    gaze_history.append((left_iris + right_iris) / 2.0)
    gaze_score = float(np.clip(
        np.std(np.array(gaze_history), axis=0).mean() / 5.0, 0.0, 1.0
    )) if len(gaze_history) >= 3 else 0.0

    # ── Signal 4: Head Motion ────────────────────────────────────────────────
    nose   = landmarks[NOSE_TIP_IDX]
    nose_pt = np.array([nose.x * img_w, nose.y * img_h])
    nose_history.append(nose_pt)
    head_score = float(np.clip(
        np.std(np.array(nose_history), axis=0).mean() / 10.0, 0.0, 1.0
    )) if len(nose_history) >= 3 else 0.0

    # ── Signal 5: Brow Tension ───────────────────────────────────────────────
    avg_brow = (
        _brow_distance(landmarks, LEFT_BROW_IDX,  LEFT_EYE_IDX,  img_w, img_h)
        + _brow_distance(landmarks, RIGHT_BROW_IDX, RIGHT_EYE_IDX, img_w, img_h)
    ) / 2.0
    if baseline_brow_ref[0] is None and frame_count <= 30:
        baseline_brow_ref[0] = avg_brow
    bb = baseline_brow_ref[0] or 0.12
    brow_tension = float(np.clip((bb - avg_brow) / (bb + 1e-8), 0.0, 1.0))

    # ── Signal 6: Jaw Tension ────────────────────────────────────────────────
    jaw_open = _jaw_openness(landmarks, JAW_IDX, img_w, img_h)
    if baseline_jaw_ref[0] is None and frame_count <= 30:
        baseline_jaw_ref[0] = jaw_open
    bj = baseline_jaw_ref[0] or 0.05
    jaw_tension = float(np.clip((bj - jaw_open) / (bj + 1e-8), 0.0, 1.0))

    # ── Weighted score ───────────────────────────────────────────────────────
    w = PHYSIO_SIGNAL_WEIGHTS
    raw   = (w["blink_rate"] * blink_score + w["eye_openness"] * eye_open_score
             + w["gaze_jitter"] * gaze_score + w["head_motion"] * head_score
             + w["brow_tension"] * brow_tension + w["jaw_tension"] * jaw_tension)
    score = float(np.clip(raw / TOTAL_WEIGHT, 0.0, 1.0))

    signals = {
        "blink_rate":          round(blink_score,    3),
        "eye_openness":        round(eye_open_score, 3),
        "gaze_jitter":         round(gaze_score,     3),
        "head_motion":         round(head_score,     3),
        "brow_tension":        round(brow_tension,   3),
        "jaw_tension":         round(jaw_tension,    3),
        "raw_ear":             round(avg_ear,        3),
        "physio_score":        round(score,          3),
        "landmarks_detected":  True,
    }
    return score, signals


# ══════════════════════════════════════════════════════════════════════════════
#  PhysioBranch — Main class
# ══════════════════════════════════════════════════════════════════════════════
class PhysioBranch:
    """
    Physiological stress signal extractor using MediaPipe FaceMesh.
    Automatically uses the correct MediaPipe API for the installed version:
      - mediapipe < 0.10.18 : mp.solutions.face_mesh  (legacy)
      - mediapipe >= 0.10.18: mp.tasks FaceLandmarker (new)
    """

    BLINK_WINDOW  = 90
    GAZE_WINDOW   = 15
    MOTION_WINDOW = 15

    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.available    = _MP_AVAILABLE
        self._face_mesh   = None       # legacy API object
        self._landmarker  = None       # tasks API object
        self._det_conf    = min_detection_confidence
        self._trk_conf    = min_tracking_confidence

        # History buffers
        self._blink_history = deque(maxlen=self.BLINK_WINDOW)
        self._gaze_history  = deque(maxlen=self.GAZE_WINDOW)
        self._nose_history  = deque(maxlen=self.MOTION_WINDOW)

        # Mutable refs for _compute_signals_from_landmarks
        self._eye_closed_ref    = [False]
        self._baseline_brow_ref = [None]
        self._baseline_jaw_ref  = [None]
        self._frame_count       = 0

        if not self.available:
            return

        if not _USE_TASKS_API:
            # Legacy: mp.solutions.face_mesh
            self._face_mesh = _mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
        else:
            # New Tasks API
            self._init_tasks_api(min_detection_confidence, min_tracking_confidence)

    def _init_tasks_api(self, det_conf: float, trk_conf: float):
        """Initialise the new mp.tasks FaceLandmarker."""
        try:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision

            model_path = _get_tasks_model_path()
            if model_path is None:
                print("[PhysioBranch] Could not obtain Tasks model. Disabling physio branch.")
                self.available = False
                return

            base_opts = mp_python.BaseOptions(model_asset_path=model_path)
            opts = mp_vision.FaceLandmarkerOptions(
                base_options=base_opts,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
                num_faces=1,
                min_face_detection_confidence=det_conf,
                min_face_presence_confidence=det_conf,
                min_tracking_confidence=trk_conf,
                running_mode=mp_vision.RunningMode.IMAGE,
            )
            self._landmarker = mp_vision.FaceLandmarker.create_from_options(opts)
            print("[PhysioBranch] FaceLandmarker (Tasks API) ready.")
        except Exception as e:
            print(f"[PhysioBranch] Tasks API init failed: {e}")
            self.available = False

    def process_frame(self, frame_bgr: np.ndarray) -> tuple[float, dict]:
        """
        Extract 6 physiological signals from a BGR frame.
        Returns (physio_score 0–1, signals dict).
        """
        if not self.available:
            return 0.0, self._zero_signals()

        img_h, img_w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        landmarks = self._get_landmarks(frame_rgb, img_w, img_h)
        if landmarks is None:
            return 0.0, self._zero_signals()

        self._frame_count += 1
        return _compute_signals_from_landmarks(
            landmarks, img_w, img_h,
            self._blink_history, self._gaze_history, self._nose_history,
            self._eye_closed_ref,
            self._baseline_brow_ref,
            self._baseline_jaw_ref,
            self._frame_count,
        )

    def _get_landmarks(self, frame_rgb: np.ndarray, img_w: int, img_h: int):
        """
        Run face landmark detection and return a flat landmark list,
        or None if no face is detected.
        Works for both legacy and Tasks API.
        """
        if not _USE_TASKS_API:
            # Legacy mp.solutions path
            frame_rgb.flags.writeable = False
            results = self._face_mesh.process(frame_rgb)
            frame_rgb.flags.writeable = True
            if not results.multi_face_landmarks:
                return None
            return results.multi_face_landmarks[0].landmark

        else:
            # New Tasks API path
            try:
                import mediapipe as mp
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB, data=frame_rgb
                )
                result = self._landmarker.detect(mp_image)
                if not result.face_landmarks:
                    return None
                # Tasks API returns NormalizedLandmark objects — same .x/.y/.z interface
                return result.face_landmarks[0]
            except Exception as e:
                print(f"[PhysioBranch] Detection error: {e}")
                return None

    def process_single_image(self, frame_bgr: np.ndarray) -> tuple[float, dict]:
        """
        Static-image mode: only compute instantaneous signals (eye openness,
        brow tension, jaw tension). Temporal signals (blink rate, gaze jitter,
        head motion) require a video stream and are set to 0 with a note.

        Weights are renormalised across the 3 valid signals so the score is
        still meaningful on a 0–1 scale.

        Use this instead of process_frame() when input is a single photo.
        """
        if not self.available:
            return 0.0, self._zero_signals()

        img_h, img_w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        landmarks = self._get_landmarks(frame_rgb, img_w, img_h)

        if landmarks is None:
            return 0.0, self._zero_signals()

        # ── Instantaneous signals only ────────────────────────────────────────
        left_ear  = _eye_aspect_ratio(landmarks, LEFT_EYE_IDX,  img_w, img_h)
        right_ear = _eye_aspect_ratio(landmarks, RIGHT_EYE_IDX, img_w, img_h)
        avg_ear   = (left_ear + right_ear) / 2.0
        # Low EAR = squinting/tired eyes → stress signal
        eye_open_score = float(np.clip(1.0 - (avg_ear / 0.30), 0.0, 1.0))

        avg_brow = (
            _brow_distance(landmarks, LEFT_BROW_IDX,  LEFT_EYE_IDX,  img_w, img_h)
            + _brow_distance(landmarks, RIGHT_BROW_IDX, RIGHT_EYE_IDX, img_w, img_h)
        ) / 2.0
        # Use population average baseline (0.12) since we have no prior frames
        brow_tension = float(np.clip((0.12 - avg_brow) / 0.12, 0.0, 1.0))

        jaw_open = _jaw_openness(landmarks, JAW_IDX, img_w, img_h)
        # Use population average baseline (0.05)
        jaw_tension = float(np.clip((0.05 - jaw_open) / 0.05, 0.0, 1.0))

        # ── Re-normalised score (only 3 valid signals) ────────────────────────
        # Original weights: eye_openness=0.16, brow=0.08, jaw=0.08  → sum=0.32
        # Renormalise to 1.0 so the score isn't artificially deflated
        w_eye  = 0.16 / 0.32   # 0.50
        w_brow = 0.08 / 0.32   # 0.25
        w_jaw  = 0.08 / 0.32   # 0.25
        physio_score = float(np.clip(
            w_eye * eye_open_score + w_brow * brow_tension + w_jaw * jaw_tension,
            0.0, 1.0
        ))

        signals = {
            "blink_rate":          0.0,   # N/A — requires video stream
            "eye_openness":        round(eye_open_score, 3),
            "gaze_jitter":         0.0,   # N/A — requires video stream
            "head_motion":         0.0,   # N/A — requires video stream
            "brow_tension":        round(brow_tension,   3),
            "jaw_tension":         round(jaw_tension,    3),
            "raw_ear":             round(avg_ear,        3),
            "physio_score":        round(physio_score,   3),
            "landmarks_detected":  True,
            "static_image_mode":   True,  # flag so UI can show a note
        }
        return physio_score, signals

    def draw_landmarks(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Draw face mesh on frame for debug visualisation."""
        if not self.available or _USE_TASKS_API:
            # Drawing utils only supported for legacy API
            return frame_bgr

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = self._face_mesh.process(frame_rgb)
        frame_rgb.flags.writeable = True

        overlay = frame_bgr.copy()
        if results.multi_face_landmarks:
            for fl in results.multi_face_landmarks:
                _mp_drawing.draw_landmarks(
                    overlay, fl,
                    _mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=_mp_drawing.DrawingSpec(
                        color=(0, 120, 0), thickness=1
                    ),
                )
        return overlay

    def _zero_signals(self) -> dict:
        return {s: 0.0 for s in [
            "blink_rate", "eye_openness", "gaze_jitter",
            "head_motion", "brow_tension", "jaw_tension",
            "raw_ear", "physio_score",
        ]} | {"landmarks_detected": False}

    def reset_calibration(self):
        self._baseline_brow_ref[0] = None
        self._baseline_jaw_ref[0]  = None
        self._frame_count = 0
        self._blink_history.clear()
        self._gaze_history.clear()
        self._nose_history.clear()

    def close(self):
        if self._face_mesh:
            self._face_mesh.close()
        if self._landmarker:
            self._landmarker.close()


# ── Quick test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    print("=" * 55)
    print("  Physio Branch — MediaPipe Signal Extractor")
    print("=" * 55)

    if not _MP_AVAILABLE:
        print("mediapipe not installed. Run: pip install mediapipe")
        sys.exit(1)

    branch = PhysioBranch()
    if not branch.available:
        print("PhysioBranch could not initialise.")
        sys.exit(1)

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    score, signals = branch.process_frame(dummy_frame)

    print(f"\nTest with blank frame (no face):")
    print(f"  Physio score : {score}")
    print(f"  Signals      : {signals}")
    print(f"\nSignal weights (sum to {TOTAL_WEIGHT}):")
    for sig, w in PHYSIO_SIGNAL_WEIGHTS.items():
        print(f"  {sig:<16} ×{w}")
    branch.close()
    print("\nPhysioBranch OK")
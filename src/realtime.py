import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data_loader import (
    EMOTION_LABELS, STRESS_MAP, IMG_SIZE,
    EMOTION_VALENCE, CNN_WEIGHT, PHYSIO_WEIGHT,
    fused_score_to_label, compute_emotion_stress_score,
)
from model       import load_model, MODEL_PATH
from physio_branch import PhysioBranch

# ── Haar Cascade for face detection (CNN branch pre-step) ─────────────────────
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

# ── JITAI — Just-In-Time Adaptive Interventions ────────────────────────────────
JITAI_SUGGESTIONS = {
    "High": [
        "[*] Try 4-7-8 breathing: inhale 4s, hold 7s, exhale 8s",
        "[*] Drink a glass of water and step away from screen",
        "[*] Take a 5-minute walk to reset your mind",
        "[*] Listen to calming music for 3 minutes",
        "[*] Write down what's stressing you right now",
    ],
    "Medium": [
        "[~] Take a short break (5-10 minutes)",
        "[~] Look at something green for 20 seconds (20-20-20 rule)",
        "[~] Do 5 neck rolls and shoulder shrugs",
        "[~] Silence notifications for the next 30 minutes",
    ],
    "Low": [
        "[+] Great focus! Keep going!",
        "[+] You're in a calm state -- ideal for creative work",
        "[+] Perfect time to tackle challenging tasks",
    ],
}

# Color coding BGR for OpenCV: Red=High, Orange=Medium, Green=Low
STRESS_COLORS = {
    "High":   (0,   0,   220),
    "Medium": (0,   165, 255),
    "Low":    (0,   200, 0),
}


# ── Display helpers ────────────────────────────────────────────────────────────
def draw_label(frame, text, pos, color=(255, 255, 255),
               font_scale=0.7, thickness=2):
    """Draw text with dark background for readability."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = pos
    cv2.rectangle(frame, (x - 2, y - th - 4), (x + tw + 2, y + 4), (0, 0, 0), -1)
    cv2.putText(frame, text, pos, font, font_scale, color, thickness)


def draw_bar(frame, value, max_val, x, y, w=150, h=18,
             color=(0, 200, 0), label=""):
    """Draw a horizontal progress bar."""
    filled = int(w * min(value / (max_val + 1e-8), 1.0))
    cv2.rectangle(frame, (x, y), (x + w, y + h), (60, 60, 60), -1)
    cv2.rectangle(frame, (x, y), (x + filled, y + h), color, -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (200, 200, 200), 1)
    if label:
        cv2.putText(frame, label, (x + w + 5, y + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1)


def draw_physio_panel(frame, signals: dict, x: int = 10, y_start: int = 80):
    """Draw the 6 physiological signal bars (debug panel)."""
    titles = {
        "blink_rate":   "Blink Rate  ×0.25",
        "eye_openness": "Eye Open    ×0.16",
        "gaze_jitter":  "Gaze Jitter ×0.16",
        "head_motion":  "Head Motion ×0.12",
        "brow_tension": "Brow Tens.  ×0.08",
        "jaw_tension":  "Jaw Tens.   ×0.08",
    }
    cv2.rectangle(frame, (x - 5, y_start - 20),
                  (x + 220, y_start + len(titles) * 26 + 10), (20, 20, 20), -1)
    cv2.putText(frame, "Physio Branch (80%)", (x, y_start - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 200, 255), 1)

    for i, (key, title) in enumerate(titles.items()):
        val = signals.get(key, 0.0)
        ry  = y_start + i * 26
        # Color: green→yellow→red based on value
        r = int(255 * val)
        g = int(255 * (1 - val))
        bar_color = (0, g, r)
        draw_bar(frame, val, 1.0, x, ry, w=80, h=14, color=bar_color)
        cv2.putText(frame, f"{title} {val:.2f}", (x + 88, ry + 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)


def preprocess_face(face_gray: np.ndarray) -> np.ndarray:
    """
    Prepare detected face for CNN inference.
    Returns: (1, 48, 48, 1) float32 array
    """
    face = cv2.resize(face_gray, (IMG_SIZE, IMG_SIZE))
    face = face.astype(np.float32) / 255.0
    return face[np.newaxis, ..., np.newaxis]


# ── Score Fuser (architecture diagram: Weighted Score Fuser) ──────────────────
def fuse_scores(emotion_score: float, physio_score: float) -> tuple[float, str, int]:
    """
    Weighted fusion: 0.20 × emotion_score + 0.80 × physio_score

    Parameters
    ----------
    emotion_score : float  0.0–1.0  (CNN branch output)
    physio_score  : float  0.0–1.0  (Physio branch output)

    Returns
    -------
    fused_score   : float  0.0–1.0
    stress_label  : str    'High' | 'Medium' | 'Low'
    stress_numeric: int    2      | 1        | 0
    """
    fused = CNN_WEIGHT * emotion_score + PHYSIO_WEIGHT * physio_score
    fused = float(np.clip(fused, 0.0, 1.0))
    label, numeric = fused_score_to_label(fused)
    return fused, label, numeric


# ── JITAI Engine ──────────────────────────────────────────────────────────────
class JITAIEngine:
    """
    Rule-based adaptive suggestion system.
    Tracks recent stress history to avoid spamming suggestions.
    """
    def __init__(self, window: int = 30, threshold: int = 20):
        self.history   = []
        self.window    = window
        self.threshold = threshold
        self._idx = {"High": 0, "Medium": 0, "Low": 0}

    def update(self, stress_level: str) -> str | None:
        """Add observation; return suggestion string if threshold reached."""
        self.history.append(stress_level)
        if len(self.history) > self.window:
            self.history.pop(0)

        if len(self.history) < self.window:
            return None

        counts   = {l: self.history.count(l) for l in ["High", "Medium", "Low"]}
        dominant = max(counts, key=counts.get)

        if counts[dominant] >= self.threshold:
            suggestions = JITAI_SUGGESTIONS[dominant]
            suggestion  = suggestions[self._idx[dominant] % len(suggestions)]
            self._idx[dominant] = (self._idx[dominant] + 1) % len(suggestions)
            self.history.clear()
            return suggestion

        return None


# ══════════════════════════════════════════════════════════════════════════════
#  Main real-time loop
# ══════════════════════════════════════════════════════════════════════════════
def run_realtime(model_path=MODEL_PATH, camera_id: int = 0):
    """
    Start dual-branch real-time stress detection.

    Keys:
      Q — quit
      S — manual suggestion
      D — toggle debug / physio panel
    """
    # ── Load CNN model ─────────────────────────────────────────────────────────
    if not Path(model_path).exists():
        print(f"CNN model not found at {model_path}")
        print("Run 03_model_training.ipynb first!")
        return

    print("Loading CNN branch model...")
    cnn_model = load_model(model_path)
    print("CNN model loaded")

    # ── Initialize Physio branch ───────────────────────────────────────────────
    print("Initializing Physio branch (MediaPipe)...")
    physio = PhysioBranch()
    if physio.available:
        print("MediaPipe FaceMesh ready (478 landmarks + iris)")
    else:
        print(" MediaPipe not available — physio score will be 0")
        print("   pip install mediapipe")

    # ── Open webcam ────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Cannot open webcam {camera_id}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print(f"Webcam opened (Q=Quit, S=Suggestion, D=Debug)")

    jitai              = JITAIEngine()
    current_suggestion = "Warming up..."
    fps_counter        = 0
    fps_start          = time.time()
    fps_display        = 0.0
    show_debug         = False

    # State for smooth display
    fused_score     = 0.0
    emotion_score   = 0.0
    physio_score_v  = 0.0
    stress_label    = "Low"
    stress_color    = STRESS_COLORS["Low"]
    current_emotion = "—"
    physio_signals  = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            print(" Frame capture failed")
            break

        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ══════════════════════════════════════════════════════════════════════
        #  PHYSIO BRANCH (80%) — runs on every frame via MediaPipe
        # ══════════════════════════════════════════════════════════════════════
        physio_score_v, physio_signals = physio.process_frame(frame)

        # ══════════════════════════════════════════════════════════════════════
        #  CNN BRANCH (20%) — Haar → face ROI → CNN emotion → valence
        # ══════════════════════════════════════════════════════════════════════
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        for (fx, fy, fw, fh) in faces:
            face_roi   = gray[fy:fy+fh, fx:fx+fw]
            face_input = preprocess_face(face_roi)

            preds   = cnn_model.predict(face_input, verbose=0)[0]
            cls     = int(np.argmax(preds))
            conf    = float(preds[cls])
            current_emotion = EMOTION_LABELS[cls]

            # CNN branch: emotion → stress valence score
            emotion_score = compute_emotion_stress_score(cls, conf)

            # ════════════════════════════════════════════════════════════════
            #  WEIGHTED SCORE FUSER
            #  0.20 × emotion_score + 0.80 × physio_score
            # ════════════════════════════════════════════════════════════════
            fused_score, stress_label, _ = fuse_scores(emotion_score, physio_score_v)
            stress_color = STRESS_COLORS[stress_label]

            # Draw face bounding box
            cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), stress_color, 2)
            draw_label(frame, f"{current_emotion}  ({conf*100:.0f}%)",
                       (fx, fy - 10), color=stress_color)
            draw_label(frame, f"Stress: {stress_label}",
                       (fx, fy + fh + 20), color=stress_color, font_scale=0.8)

            # Confidence bar
            draw_bar(frame, conf, 1.0, fx, fy + fh + 38,
                     w=min(fw, 140), h=12, color=stress_color)

            # JITAI
            suggestion = jitai.update(stress_label)
            if suggestion:
                current_suggestion = suggestion

        # ══════════════════════════════════════════════════════════════════════
        #  HUD — scores & labels
        # ══════════════════════════════════════════════════════════════════════
        # Score panel (top-right area)
        panel_x = w - 250
        cv2.rectangle(frame, (panel_x - 8, 5), (w - 5, 135), (20, 20, 20), -1)
        cv2.putText(frame, "Score Fuser", (panel_x, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 180, 255), 1)

        draw_label(frame, f"CNN(20%) emotion: {emotion_score:.2f}",
                   (panel_x, 42), font_scale=0.45, thickness=1,
                   color=(150, 220, 255))
        draw_label(frame, f"Physio(80%) score: {physio_score_v:.2f}",
                   (panel_x, 62), font_scale=0.45, thickness=1,
                   color=(150, 255, 150))
        draw_label(frame, f"Fused = {fused_score*100:.0f}/100",
                   (panel_x, 82), font_scale=0.50, thickness=1,
                   color=stress_color)

        # Fused score bar
        draw_bar(frame, fused_score, 1.0, panel_x, 90,
                 w=180, h=16, color=stress_color, label=stress_label)

        # Stress thresholds label
        cv2.putText(frame, "L:0-33 M:34-66 H:67-100",
                    (panel_x, 122), cv2.FONT_HERSHEY_SIMPLEX,
                    0.38, (160, 160, 160), 1)

        # ── FPS ───────────────────────────────────────────────────────────────
        fps_counter += 1
        if fps_counter % 30 == 0:
            elapsed     = time.time() - fps_start
            fps_display = 30 / (elapsed + 1e-8)
            fps_start   = time.time()
        draw_label(frame, f"FPS: {fps_display:.1f}", (10, 25),
                   font_scale=0.55, thickness=1)

        # ── Suggestion banner ─────────────────────────────────────────────────
        if current_suggestion:
            draw_label(frame, f"{current_suggestion[:68]}",
                       (10, h - 35), color=(255, 220, 0),
                       font_scale=0.50, thickness=1)

        draw_label(frame, "Q=Quit  S=Suggest  D=Debug",
                   (10, h - 10), font_scale=0.45, thickness=1)

        if len(faces) == 0:
            draw_label(frame, "No face detected — look at camera",
                       (w // 2 - 170, h // 2),
                       color=(100, 100, 255), font_scale=0.65)

        # ── Debug panel (physio signals) ──────────────────────────────────────
        if show_debug and physio_signals:
            draw_physio_panel(frame, physio_signals, x=10, y_start=80)

        cv2.imshow("Dual-Branch Stress Detection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("Exiting...")
            break
        elif key == ord("s"):
            current_suggestion = JITAI_SUGGESTIONS["High"][0]
        elif key == ord("d"):
            show_debug = not show_debug
            print(f"Debug panel: {'ON' if show_debug else 'OFF'}")

    cap.release()
    cv2.destroyAllWindows()
    physio.close()
    print("Resources released")


if __name__ == "__main__":
    run_realtime()
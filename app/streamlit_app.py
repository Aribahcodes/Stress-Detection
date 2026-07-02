"""
app/streamlit_app.py — Dual-Branch Stress Detection Web App
============================================================
ARCHITECTURE:
  CNN Branch (20%)   : FER2013 CNN → 7-class softmax → stress valence
  Physio Branch (80%): MediaPipe 478 pts → 6 signals → physio score
  Fuser              : 0.20 × emotion + 0.80 × physio → fused score
  Output             : Low(0-33) | Medium(34-66) | High(67-100)
  JITAI              : Rule-based adaptive suggestions

Run with:
    streamlit run app/streamlit_app.py

Requires (in addition to existing requirements):
    pip install streamlit-webrtc aiortc
"""

import sys
import io
import random
import threading
import numpy as np
import cv2
import streamlit as st
from pathlib import Path
from collections import deque

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data_loader import (
    EMOTION_LABELS, STRESS_MAP, IMG_SIZE,
    CNN_WEIGHT, PHYSIO_WEIGHT,
    compute_emotion_stress_score, fused_score_to_label,
)
from realtime import (
    preprocess_face, JITAI_SUGGESTIONS, STRESS_COLORS,
    face_cascade, fuse_scores, draw_label, draw_bar, draw_physio_panel,
    JITAIEngine,
)
from physio_branch import PhysioBranch

# ── streamlit-webrtc ──────────────────────────────────────────────────────────
try:
    from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Dual-Branch Stress Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  .stress-high   { background:#ff4b4b22; border-left:4px solid #ff4b4b;
                   padding:12px; border-radius:6px; margin:6px 0; }
  .stress-medium { background:#ffa50022; border-left:4px solid #ffa500;
                   padding:12px; border-radius:6px; margin:6px 0; }
  .stress-low    { background:#00c80022; border-left:4px solid #00c800;
                   padding:12px; border-radius:6px; margin:6px 0; }
  .branch-card   { background:#1e1e2e; border-radius:10px;
                   padding:14px; margin:6px 0; }
  .suggestion-box{ background:#2a2a3e; border-radius:10px;
                   padding:16px; border:1px solid #555; margin:6px 0; }
  .fuser-box     { background:#2d1f3e; border:2px solid #8855ff;
                   border-radius:10px; padding:14px; margin:6px 0; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  SHARED STATE  (thread-safe — written by WebRTC callback, read by UI)
# ══════════════════════════════════════════════════════════════════════════════
class SharedState:
    """Thread-safe container for results produced by the video callback."""
    def __init__(self):
        self._lock          = threading.Lock()
        self.emotion        = "—"
        self.stress         = "—"
        self.confidence     = 0.0
        self.emotion_score  = 0.0
        self.physio_score   = 0.0
        self.fused_score    = 0.0
        self.physio_signals = {}
        self.suggestion     = ""
        self.frames         = 0
        self.stress_history  = deque(maxlen=120)
        self.fused_history   = deque(maxlen=120)
        self.emotion_history = deque(maxlen=120)

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "emotion":        self.emotion,
                "stress":         self.stress,
                "confidence":     self.confidence,
                "emotion_score":  self.emotion_score,
                "physio_score":   self.physio_score,
                "fused_score":    self.fused_score,
                "physio_signals": dict(self.physio_signals),
                "suggestion":     self.suggestion,
                "frames":         self.frames,
                "stress_history": list(self.stress_history),
                "fused_history":  list(self.fused_history),
                "emotion_history":list(self.emotion_history),
            }


if "shared"       not in st.session_state:
    st.session_state.shared       = SharedState()
if "jitai_engine" not in st.session_state:
    st.session_state.jitai_engine = JITAIEngine(window=30, threshold=20)
if "show_debug"   not in st.session_state:
    st.session_state.show_debug   = True


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL / PHYSIO BRANCH  (cached — loaded once per session)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_cnn_model():
    model_path = PROJECT_ROOT / "models" / "stress_emotion_model.h5"
    if not model_path.exists():
        return None
    import tensorflow as tf
    from tensorflow import keras
    return keras.models.load_model(str(model_path))


@st.cache_resource
def load_physio_branch():
    return PhysioBranch()


cnn_model = load_cnn_model()
physio    = load_physio_branch()


# ══════════════════════════════════════════════════════════════════════════════
#  VIDEO FRAME CALLBACK  (runs in WebRTC thread — NOT the Streamlit main thread)
# ══════════════════════════════════════════════════════════════════════════════
def make_video_callback(cnn_model, physio: PhysioBranch,
                        shared: SharedState, jitai: JITAIEngine,
                        show_debug: bool):
    """Factory — binds dependencies into a closure for streamlit-webrtc."""

    def callback(frame):
        import av
        img  = frame.to_ndarray(format="bgr24")
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # ── Physio Branch (80%) — all 6 signals active ────────────────────────
        physio_score, physio_signals = physio.process_frame(img)

        # ── CNN Branch (20%) ──────────────────────────────────────────────────
        emotion       = "—"
        emotion_score = 0.0
        fused_score   = float(np.clip(PHYSIO_WEIGHT * physio_score, 0.0, 1.0))
        stress_label  = fused_score_to_label(fused_score)[0]
        conf          = 0.0

        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        for (fx, fy, fw, fh) in faces:
            face_roi   = gray[fy:fy+fh, fx:fx+fw]
            face_input = preprocess_face(face_roi)
            preds      = cnn_model.predict(face_input, verbose=0)[0]
            cls        = int(np.argmax(preds))
            conf       = float(preds[cls])
            emotion    = EMOTION_LABELS[cls]
            emotion_score            = compute_emotion_stress_score(cls, conf)
            fused_score, stress_label, _ = fuse_scores(emotion_score, physio_score)
            color_bgr = STRESS_COLORS[stress_label]

            cv2.rectangle(img, (fx, fy), (fx+fw, fy+fh), color_bgr, 2)
            draw_label(img, f"{emotion} ({conf*100:.0f}%)",
                       (fx, fy - 10), color=color_bgr)
            draw_label(img, f"Stress: {stress_label}",
                       (fx, fy + fh + 22), color=color_bgr, font_scale=0.75)
            draw_bar(img, conf, 1.0, fx, fy + fh + 38,
                     w=min(fw, 130), h=10, color=color_bgr)

        # ── Score panel (top-right) ───────────────────────────────────────────
        px = w - 255
        cv2.rectangle(img, (px - 8, 4), (w - 4, 130), (20, 20, 20), -1)
        cv2.putText(img, "Score Fuser", (px, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 255), 1)
        draw_label(img, f"CNN (20%)  emotion: {emotion_score:.3f}",
                   (px, 38), font_scale=0.42, thickness=1, color=(150, 220, 255))
        draw_label(img, f"Physio (80%) score: {physio_score:.3f}",
                   (px, 56), font_scale=0.42, thickness=1, color=(150, 255, 150))
        stress_color = STRESS_COLORS.get(stress_label, (200, 200, 200))
        draw_label(img, f"Fused = {fused_score*100:.0f}/100  [{stress_label}]",
                   (px, 74), font_scale=0.48, thickness=1, color=stress_color)
        draw_bar(img, fused_score, 1.0, px, 82, w=200, h=16, color=stress_color)
        cv2.putText(img, "L:0-33  M:34-66  H:67-100",
                    (px, 116), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (140, 140, 140), 1)

        # ── JITAI ─────────────────────────────────────────────────────────────
        suggestion = jitai.update(stress_label)
        if suggestion:
            with shared._lock:
                shared.suggestion = suggestion
        with shared._lock:
            current_suggestion = shared.suggestion
        if current_suggestion:
            draw_label(img, current_suggestion[:72],
                       (10, h - 30), color=(255, 220, 0), font_scale=0.46)

        # ── No-face warning ───────────────────────────────────────────────────
        if len(faces) == 0:
            draw_label(img, "No face detected — look at camera",
                       (w // 2 - 180, h // 2), color=(100, 100, 255))

        # ── Physio debug bars ─────────────────────────────────────────────────
        if show_debug and physio_signals:
            draw_physio_panel(img, physio_signals, x=10, y_start=60)

        # ── LIVE badge ────────────────────────────────────────────────────────
        cv2.circle(img, (16, 16), 7, (0, 0, 220), -1)
        cv2.putText(img, "LIVE", (27, 21),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 220), 1)

        # ── Update shared state ───────────────────────────────────────────────
        with shared._lock:
            shared.emotion        = emotion
            shared.stress         = stress_label
            shared.confidence     = conf
            shared.emotion_score  = emotion_score
            shared.physio_score   = physio_score
            shared.fused_score    = fused_score
            shared.physio_signals = physio_signals
            shared.frames        += 1
            shared.stress_history.append(stress_label)
            shared.fused_history.append(round(fused_score, 3))
            shared.emotion_history.append(emotion)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

    return callback


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("⚙️  Controls")
    st.markdown("---")

    if cnn_model:
        st.success("✅ CNN Model loaded")
    else:
        st.error("❌ CNN model not found\n`models/stress_emotion_model.keras`")

    if physio and physio.available:
        st.success("✅ Physio Branch (MediaPipe) ready")
    else:
        st.warning("⚠️ MediaPipe not available\n`pip install mediapipe`")

    st.markdown("---")
    st.subheader("Fusion Weights")
    st.markdown(f"""
| Branch | Weight |
|--------|--------|
| CNN (emotion) | **{CNN_WEIGHT*100:.0f}%** |
| Physio (MediaPipe) | **{PHYSIO_WEIGHT*100:.0f}%** |
""")

    st.markdown("---")
    st.session_state.show_debug = st.checkbox(
        "Show physio signal bars on video", value=st.session_state.show_debug
    )

    st.markdown("---")
    st.subheader("Live Session Stats")
    snap = st.session_state.shared.snapshot()
    st.metric("Frames Processed", snap["frames"])
    st.metric("Current Emotion",  snap["emotion"])
    st.metric("Stress Level",     snap["stress"])
    st.metric("Fused Score",      f"{snap['fused_score']*100:.0f}/100")

    st.markdown("---")
    if st.button("Clear History", use_container_width=True):
        s = st.session_state.shared
        with s._lock:
            s.stress_history.clear()
            s.fused_history.clear()
            s.emotion_history.clear()
            s.frames = 0
        st.rerun()

    st.caption("Dual-Branch Stress Detection v2.0\nCNN(20%) + MediaPipe(80%)")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN LAYOUT
# ══════════════════════════════════════════════════════════════════════════════
st.title("🧠 Dual-Branch Stress Detection & Management System")
st.caption(
    "CNN Branch (20%): FER2013 emotion → stress valence  |  "
    "Physio Branch (80%): MediaPipe 478 landmarks → 6 physiological signals"
)

tab1, tab2, tab3, tab4 = st.tabs(
    ["📹 Live Detection", "📷 Single Image", "📈 History", "🔬 Architecture"]
)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — LIVE VIDEO
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    if not WEBRTC_AVAILABLE:
        st.error(
            "**streamlit-webrtc is not installed.**\n\n"
            "Install it and restart the app:\n"
            "```\npip install streamlit-webrtc aiortc\n```"
        )
    elif cnn_model is None:
        st.warning("CNN model not loaded — check `models/stress_emotion_model.keras`.")
    else:
        col_video, col_info = st.columns([3, 2])

        with col_video:
            st.subheader("📹 Live Webcam Feed")
            st.caption(
                "All 6 physio signals are active in live mode. "
                "Blink rate and gaze jitter warm up over the first ~90 frames (~3 s)."
            )

            RTC_CONFIG = RTCConfiguration(
                {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
            )

            ctx = webrtc_streamer(
                key="stress-detection",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=RTC_CONFIG,
                video_frame_callback=make_video_callback(
                    cnn_model,
                    physio,
                    st.session_state.shared,
                    st.session_state.jitai_engine,
                    st.session_state.show_debug,
                ),
                media_stream_constraints={
                    "video": {"width": {"ideal": 640}, "height": {"ideal": 480}},
                    "audio": False,
                },
                async_processing=True,
            )

            if ctx.state.playing:
                st.success("● LIVE — processing every frame")
            else:
                st.info("Click **START** to begin live dual-branch detection.")

        with col_info:
            st.subheader("Live Results")
            snap = st.session_state.shared.snapshot()

            st.markdown(f"""
<div class="branch-card">
  🔵 <b>CNN Branch (20%)</b><br>
  Emotion: <b>{snap['emotion']}</b> &nbsp; Conf: {snap['confidence']*100:.1f}%<br>
  Stress score: <code>{snap['emotion_score']:.3f}</code>
</div>
""", unsafe_allow_html=True)

            sig = snap["physio_signals"]
            if sig:
                st.markdown(f"""
<div class="branch-card">
  🟢 <b>Physio Branch (80%)</b> — all 6 signals live<br>
  Blink rate: {sig.get('blink_rate',0):.2f} &nbsp;
  Eye open: {sig.get('eye_openness',0):.2f}<br>
  Gaze jitter: {sig.get('gaze_jitter',0):.2f} &nbsp;
  Head motion: {sig.get('head_motion',0):.2f}<br>
  Brow tension: {sig.get('brow_tension',0):.2f} &nbsp;
  Jaw tension: {sig.get('jaw_tension',0):.2f}<br>
  Physio score: <code>{snap['physio_score']:.3f}</code>
</div>
""", unsafe_allow_html=True)

            es = snap['emotion_score']
            ps = snap['physio_score']
            fs = snap['fused_score']
            st.markdown(f"""
<div class="fuser-box">
  <b>Weighted Score Fuser</b><br>
  <code>0.20 × {es:.3f} + 0.80 × {ps:.3f} = {fs:.3f}</code><br>
  Fused score: <b>{fs*100:.0f} / 100</b>
</div>
""", unsafe_allow_html=True)

            s = snap["stress"]
            if s == "High":
                st.markdown(
                    f'<div class="stress-high">🔴 <b>HIGH STRESS</b> (67–100)<br>'
                    f'Fused: {fs*100:.0f}/100</div>', unsafe_allow_html=True)
            elif s == "Medium":
                st.markdown(
                    f'<div class="stress-medium">🟠 <b>MEDIUM STRESS</b> (34–66)<br>'
                    f'Fused: {fs*100:.0f}/100</div>', unsafe_allow_html=True)
            elif s == "Low":
                st.markdown(
                    f'<div class="stress-low">🟢 <b>LOW STRESS</b> (0–33)<br>'
                    f'Fused: {fs*100:.0f}/100</div>', unsafe_allow_html=True)
            else:
                st.info("Start the camera to see results.")

            if snap["suggestion"]:
                st.markdown("---")
                st.subheader("💡 JITAI Suggestion")
                st.markdown(
                    f'<div class="suggestion-box">{snap["suggestion"]}</div>',
                    unsafe_allow_html=True)

            if snap["fused_history"]:
                st.markdown("---")
                st.caption("Fused stress score — last 120 frames")
                import pandas as pd
                st.line_chart(
                    pd.DataFrame({"Fused Score": snap["fused_history"]}),
                    color="#8855ff", height=130,
                )


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — SINGLE IMAGE
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    col_cam, col_info2 = st.columns([3, 2])

    with col_cam:
        st.subheader("Image Input")
        st.caption(
            "⚠️ Static image mode: blink rate, gaze jitter and head motion "
            "require a video stream and are excluded here. "
            "Use **Live Detection** for full accuracy."
        )
        uploaded     = st.file_uploader(
            "Upload an image", type=["jpg", "jpeg", "png"], key="face_upload"
        )
        camera_image = st.camera_input("Or take a snapshot")
        analyze_btn  = st.button("Analyze (Static Image)", type="primary",
                                  use_container_width=True)

        if analyze_btn:
            img_source = camera_image or uploaded
            if img_source is None:
                st.warning("Please take a snapshot or upload an image first.")
            elif cnn_model is None:
                st.error("CNN model not found.")
            else:
                from PIL import Image as PILImage
                img_bytes = img_source.read()
                pil_img   = PILImage.open(io.BytesIO(img_bytes)).convert("RGB")
                frame_rgb = np.array(pil_img)
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                gray      = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

                if physio and physio.available:
                    physio_score, physio_signals = physio.process_single_image(frame_bgr)
                else:
                    physio_score, physio_signals = 0.0, {}

                faces = face_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
                )
                if len(faces) == 0:
                    st.warning("No face detected. Try better lighting or a closer shot.")
                else:
                    for (fx, fy, fw, fh) in faces:
                        face_roi   = gray[fy:fy+fh, fx:fx+fw]
                        face_input = preprocess_face(face_roi)
                        preds      = cnn_model.predict(face_input, verbose=0)[0]
                        cls        = int(np.argmax(preds))
                        conf       = float(preds[cls])
                        emotion    = EMOTION_LABELS[cls]
                        emotion_score = compute_emotion_stress_score(cls, conf)
                        fused_score, stress_label, _ = fuse_scores(emotion_score, physio_score)
                        color_bgr = STRESS_COLORS[stress_label]

                        cv2.rectangle(frame_bgr, (fx, fy), (fx+fw, fy+fh), color_bgr, 3)
                        cv2.putText(frame_bgr, f"{emotion} ({conf*100:.0f}%)",
                                    (fx, fy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color_bgr, 2)
                        cv2.putText(frame_bgr, f"Stress: {stress_label}",
                                    (fx, fy + fh + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_bgr, 2)

                        st.session_state["_s_emotion"]  = emotion
                        st.session_state["_s_stress"]   = stress_label
                        st.session_state["_s_conf"]     = conf
                        st.session_state["_s_es"]       = emotion_score
                        st.session_state["_s_ps"]       = physio_score
                        st.session_state["_s_sig"]      = physio_signals
                        st.session_state["_s_fs"]       = fused_score

                    st.image(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB),
                             caption="Detection Result", use_column_width=True)

    with col_info2:
        st.subheader("Static Results")
        e  = st.session_state.get("_s_emotion", "—")
        s  = st.session_state.get("_s_stress",  "—")
        c  = st.session_state.get("_s_conf",    0.0)
        es = st.session_state.get("_s_es",      0.0)
        ps = st.session_state.get("_s_ps",      0.0)
        fs = st.session_state.get("_s_fs",      0.0)
        sig= st.session_state.get("_s_sig",     {})

        st.markdown(f"""
<div class="branch-card">
  🔵 <b>CNN Branch (20%)</b><br>
  Emotion: <b>{e}</b> &nbsp; Conf: {c*100:.1f}%<br>
  Stress score: <code>{es:.3f}</code>
</div>
""", unsafe_allow_html=True)

        if sig:
            na = "<span style='color:#888'>N/A — video only</span>"
            st.markdown(f"""
<div class="branch-card">
  🟢 <b>Physio Branch</b> — static image (3/6 signals)<br>
  Blink rate: {na} &nbsp; Eye open: {sig.get('eye_openness',0):.2f}<br>
  Gaze jitter: {na} &nbsp; Head motion: {na}<br>
  Brow tension: {sig.get('brow_tension',0):.2f} &nbsp;
  Jaw tension: {sig.get('jaw_tension',0):.2f}<br>
  Physio score: <code>{ps:.3f}</code>
</div>
""", unsafe_allow_html=True)

        st.markdown(f"""
<div class="fuser-box">
  <b>Weighted Score Fuser</b><br>
  <code>0.20 × {es:.3f} + 0.80 × {ps:.3f} = {fs:.3f}</code><br>
  Fused score: <b>{fs*100:.0f} / 100</b>
</div>
""", unsafe_allow_html=True)

        if s == "High":
            st.markdown(f'<div class="stress-high">🔴 <b>HIGH STRESS</b><br>Fused: {fs*100:.0f}/100</div>',
                        unsafe_allow_html=True)
        elif s == "Medium":
            st.markdown(f'<div class="stress-medium">🟠 <b>MEDIUM STRESS</b><br>Fused: {fs*100:.0f}/100</div>',
                        unsafe_allow_html=True)
        elif s == "Low":
            st.markdown(f'<div class="stress-low">🟢 <b>LOW STRESS</b><br>Fused: {fs*100:.0f}/100</div>',
                        unsafe_allow_html=True)
        else:
            st.info("Analyze an image to see results.")

        st.markdown("---")
        st.markdown("""
| Emotion | Valence | Stress |
|---------|---------|--------|
| 😠 Angry / 😨 Fear / 😒 Disgust | 1.0 | 🔴 High |
| 😢 Sad / 😮 Surprise | 0.5 | 🟠 Medium |
| 😐 Neutral / 😊 Happy | 0.0 | 🟢 Low |
""")


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — HISTORY
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("📈 Session History")
    snap = st.session_state.shared.snapshot()

    if not snap["stress_history"]:
        st.info("No live data yet. Start the camera in Live Detection tab.")
    else:
        import pandas as pd
        s_map   = {"High": 2, "Medium": 1, "Low": 0}
        numeric = [s_map.get(x, 0) for x in snap["stress_history"]]
        n       = len(numeric)
        df = pd.DataFrame({
            "Frame":        list(range(1, n + 1)),
            "Stress Score": numeric,
            "Fused (0-1)":  snap["fused_history"][:n],
            "Stress Level": snap["stress_history"],
            "Emotion":      snap["emotion_history"][:n],
        })
        col_a, col_b = st.columns(2)
        with col_a:
            st.caption("Stress level (0=Low 1=Medium 2=High)")
            st.line_chart(df.set_index("Frame")["Stress Score"], color="#ff4b4b")
        with col_b:
            st.caption("Fused score (0.0–1.0)")
            st.line_chart(df.set_index("Frame")["Fused (0-1)"], color="#8855ff")
        st.dataframe(df.tail(20), use_container_width=True)
        col_x, col_y, col_z = st.columns(3)
        col_x.metric("🔴 High",   snap["stress_history"].count("High"))
        col_y.metric("🟠 Medium", snap["stress_history"].count("Medium"))
        col_z.metric("🟢 Low",    snap["stress_history"].count("Low"))


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 4 — ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("🔬 Dual-Branch System Architecture")
    st.image(r"C:\Users\ariba\Downloads\Architecture DL.png")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### CNN Branch")
        st.markdown("""
- Input: 48×48 grayscale face
- 4 conv blocks: 32→64→128→256 filters
- GlobalAvgPool → Dense(256) → Softmax(7)
- Dataset: FER2013 (35,887 images)
""")
    with col2:
        st.markdown("#### Physio Branch (live mode)")
        st.markdown("""
- MediaPipe FaceMesh — 478 landmarks + iris
- **Blink rate** (×0.25) — temporal, needs ~90 frames to warm up
- **Eye openness** (×0.16) — instantaneous
- **Gaze jitter** (×0.16) — temporal, 15-frame window
- **Head motion** (×0.12) — temporal, 15-frame window
- **Brow tension** (×0.08) — instantaneous
- **Jaw tension** (×0.08) — instantaneous
""")
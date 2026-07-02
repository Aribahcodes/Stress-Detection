# -*- coding: utf-8 -*-
"""
app/gui.py — Dual-Branch Stress Detection Desktop GUI
======================================================
Tkinter desktop app replicating all Streamlit features:
  • Live webcam feed with CV2 overlays
  • Real-time signal graphs (6 physio signals)
  • Stress history chart
  • JITAI suggestion panel
  • Settings / config panel
  • Session report export (CSV + TXT summary)

Design: dark "mission-control" aesthetic — deep navy panels,
        cyan/amber accent lines, monospaced readouts.

Run with:
    python app/gui.py
"""

import sys
import os
import csv
import json
import threading
import time
import random
import queue
from pathlib import Path
from datetime import datetime
from collections import deque

import numpy as np
import cv2
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# ── Project path setup ────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data_loader import (
    EMOTION_LABELS, IMG_SIZE,
    CNN_WEIGHT, PHYSIO_WEIGHT,
    compute_emotion_stress_score, fused_score_to_label,
)
from realtime import (
    preprocess_face, JITAI_SUGGESTIONS, STRESS_COLORS,
    face_cascade, fuse_scores, JITAIEngine,
    draw_label, draw_bar, draw_physio_panel,
)
from physio_branch import PhysioBranch

# ── Optional matplotlib for in-app charts ────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    import matplotlib.animation as animation
    MPL_AVAILABLE = True
except ImportError:
    MPL_AVAILABLE = False

# ══════════════════════════════════════════════════════════════════════════════
#  DESIGN TOKENS  (mission-control dark theme)
# ══════════════════════════════════════════════════════════════════════════════
C = {
    "bg":          "#0a0e1a",   # deep navy background
    "panel":       "#0f1628",   # panel background
    "panel2":      "#141d35",   # slightly lighter panel
    "border":      "#1e2d50",   # subtle borders
    "accent":      "#00c8ff",   # cyan accent
    "accent2":     "#ff8c00",   # amber accent
    "green":       "#00e676",   # low-stress green
    "amber":       "#ffab40",   # medium-stress amber
    "red":         "#ff4444",   # high-stress red
    "text":        "#c8d8f0",   # primary text
    "text_dim":    "#5a6a8a",   # dimmed text
    "text_bright": "#ffffff",   # bright text
    "mono":        "Courier New",   # monospaced font
    "sans":        "TkDefaultFont",
}

FONT_TITLE   = ("Courier New", 13, "bold")
FONT_LABEL   = ("Courier New", 10)
FONT_SMALL   = ("Courier New",  9)
FONT_BIG     = ("Courier New", 22, "bold")
FONT_MED     = ("Courier New", 15, "bold")
FONT_SECTION = ("Courier New", 11, "bold")

SIGNAL_COLORS = {
    "blink_rate":   "#00c8ff",
    "eye_openness": "#00e676",
    "gaze_jitter":  "#ffab40",
    "head_motion":  "#ff8c00",
    "brow_tension": "#e040fb",
    "jaw_tension":  "#ff4444",
}

HISTORY_LEN = 150   # frames kept in rolling charts


# ══════════════════════════════════════════════════════════════════════════════
#  DETECTION THREAD
# ══════════════════════════════════════════════════════════════════════════════
class DetectionThread(threading.Thread):
    """
    Runs in background. Reads webcam → CNN + Physio → puts results on queue.
    The GUI thread only reads from the queue — no blocking.
    """

    def __init__(self, result_queue: queue.Queue, settings: dict,
                 cnn_model, physio: PhysioBranch, jitai: JITAIEngine):
        super().__init__(daemon=True)
        self.q        = result_queue
        self.settings = settings
        self.model    = cnn_model
        self.physio   = physio
        self.jitai    = jitai
        self._stop    = threading.Event()
        self.current_frame = None
        self._frame_lock   = threading.Lock()

    def stop(self):
        self._stop.set()

    def get_frame(self):
        with self._frame_lock:
            return self.current_frame.copy() if self.current_frame is not None else None

    def run(self):
        cam_id = self.settings.get("camera_id", 0)

        cap = None
        candidates = [cam_id] + [i for i in [0, 1, 2, 700, 701] if i != cam_id]

        for idx in candidates:
            backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
            for backend in backends:
                try:
                    test = cv2.VideoCapture(idx, backend)
                except Exception:
                    continue
                if test.isOpened():
                    ret, _ = test.read()
                    if ret:
                        cap = test
                        cam_id = idx
                        break
                    test.release()
                else:
                    test.release()
            if cap is not None:
                break

        if cap is None or not cap.isOpened():
            self.q.put({
                "error": (
                    f"Cannot open any camera (tried: {candidates[:4]}).\n\n"
                    "Then change Camera ID in the Settings tab and try again."
                )
            })
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

        fps_t        = time.time()
        fps_c        = 0
        fps          = 0.0
        frame_count  = 0

        # CNN prediction state — updated every N frames to reduce lag
        CNN_EVERY    = 5        # run CNN once every 5 frames (~6fps on slow CPU)
        last_emotion       = "—"
        last_conf          = 0.0
        last_emotion_score = 0.0
        last_fused_score   = 0.0
        last_stress_label  = "Low"
        last_physio_score  = 0.0
        last_physio_signals = {}
        last_face_box      = None   # (fx, fy, fw, fh)

        while not self._stop.is_set():
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.02)
                continue

            frame_count += 1
            h, w = frame.shape[:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # ── Physio branch — every frame (fast, no model inference) ────────
            physio_score, physio_signals = self.physio.process_frame(frame)
            last_physio_score   = physio_score
            last_physio_signals = physio_signals

            # ── CNN branch — every CNN_EVERY frames only ──────────────────────
            if frame_count % CNN_EVERY == 0:
                faces = face_cascade.detectMultiScale(
                    gray, scaleFactor=1.1,
                    minNeighbors=self.settings.get("min_neighbors", 5),
                    minSize=(30, 30)
                )
                if len(faces) > 0:
                    fx, fy, fw, fh = faces[0]
                    last_face_box  = (fx, fy, fw, fh)
                    face_roi       = gray[fy:fy+fh, fx:fx+fw]
                    face_input     = preprocess_face(face_roi)
                    preds          = self.model(face_input, training=False).numpy()[0]
                    cls            = int(np.argmax(preds))
                    last_conf      = float(preds[cls])
                    last_emotion   = EMOTION_LABELS[cls]
                    last_emotion_score = compute_emotion_stress_score(cls, last_conf)
                    last_fused_score, last_stress_label, _ = fuse_scores(
                        last_emotion_score, physio_score
                    )
                else:
                    last_face_box = None

            # ── Draw on frame — emotion box always, no stress overlay ─────────
            if last_face_box is not None:
                fx, fy, fw, fh = last_face_box
                # Neutral cyan box — no stress colour on video
                cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), (0, 200, 200), 2)
                draw_label(frame,
                           f"{last_emotion}  ({last_conf*100:.0f}%)",
                           (fx, fy - 10),
                           color=(0, 200, 200))
            else:
                draw_label(frame, "No face detected",
                           (w // 2 - 110, h // 2),
                           color=(80, 80, 220), font_scale=0.65)

            # ── FPS counter ───────────────────────────────────────────────────
            fps_c += 1
            if fps_c >= 15:
                fps   = fps_c / (time.time() - fps_t + 1e-8)
                fps_t = time.time()
                fps_c = 0
            draw_label(frame, f"FPS {fps:.1f}", (10, 22),
                       font_scale=0.48, thickness=1, color=(120, 120, 120))

            # ── REC dot ───────────────────────────────────────────────────────
            cv2.circle(frame, (w - 20, h - 15), 6, (0, 0, 200), -1)
            cv2.putText(frame, "REC", (w - 50, h - 9),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 200), 1)

            # Store frame
            with self._frame_lock:
                self.current_frame = frame.copy()

            # Push result to GUI queue
            suggestion = self.jitai.update(last_stress_label)
            result = {
                "emotion":        last_emotion,
                "stress":         last_stress_label,
                "confidence":     last_conf,
                "emotion_score":  last_emotion_score,
                "physio_score":   last_physio_score,
                "fused_score":    last_fused_score,
                "physio_signals": last_physio_signals,
                "suggestion":     suggestion,
                "fps":            fps,
                "timestamp":      datetime.now().isoformat(),
            }
            try:
                self.q.put_nowait(result)
            except queue.Full:
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    pass
                self.q.put_nowait(result)

        cap.release()


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN GUI APPLICATION
# ══════════════════════════════════════════════════════════════════════════════
class StressDetectionApp(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("Dual-Branch Stress Detection  --  Mission Control")
        self.geometry("1400x860")
        self.minsize(1100, 700)
        self.configure(bg=C["bg"])

        # State
        self._running        = False
        self._thread         = None
        self._result_queue   = queue.Queue(maxsize=5)
        self._cnn_model      = None
        self._physio         = None
        self._jitai          = JITAIEngine(window=30, threshold=20)
        self._current_suggestion = ""
        self._session_log    = []   # list of result dicts

        # Rolling history deques
        self._hist_fused   = deque(maxlen=HISTORY_LEN)
        self._hist_stress  = deque(maxlen=HISTORY_LEN)  # 0/1/2
        self._hist_signals = {k: deque(maxlen=HISTORY_LEN) for k in SIGNAL_COLORS}
        self._hist_emotion = deque(maxlen=HISTORY_LEN)

        # 5-second capture state
        self._capture_mode   = False   # True while counting down
        self._capture_frames = []      # collected results during 5s window
        self._capture_end    = 0.0     # time.time() when capture ends
        self._countdown_var  = tk.StringVar(value="")

        # Settings (mutable dict shared with thread)
        self._settings = {
            "camera_id":    0,
            "show_overlay": False,
            "show_debug":   False,
            "min_neighbors": 5,
            "cnn_weight":   CNN_WEIGHT,
            "physio_weight": PHYSIO_WEIGHT,
        }

        self._build_ui()
        self._load_model_and_physio()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── UI BUILD ──────────────────────────────────────────────────────────────

    def _build_ui(self):
        """Assemble the full window layout."""
        # ── Title bar ─────────────────────────────────────────────────────────
        title_bar = tk.Frame(self, bg=C["bg"], height=48)
        title_bar.pack(fill="x", padx=0, pady=0)
        title_bar.pack_propagate(False)

        tk.Label(title_bar,
                 text="[*]  DUAL-BRANCH STRESS DETECTION  --  MISSION CONTROL",
                 font=("Courier New", 13, "bold"),
                 fg=C["accent"], bg=C["bg"]).pack(side="left", padx=18, pady=10)

        self._status_var = tk.StringVar(value="[--] OFFLINE")
        self._status_lbl = tk.Label(title_bar, textvariable=self._status_var,
                                    font=FONT_LABEL, fg=C["text_dim"], bg=C["bg"])
        self._status_lbl.pack(side="right", padx=18)

        tk.Frame(self, bg=C["border"], height=1).pack(fill="x")

        # ── Notebook (tabs) ───────────────────────────────────────────────────
        style = ttk.Style(self)
        style.theme_use("default")
        style.configure("TNotebook",
                        background=C["bg"], borderwidth=0, tabmargins=[0, 0, 0, 0])
        style.configure("TNotebook.Tab",
                        background=C["panel"], foreground=C["text_dim"],
                        font=FONT_LABEL, padding=[18, 6],
                        borderwidth=0)
        style.map("TNotebook.Tab",
                  background=[("selected", C["panel2"])],
                  foreground=[("selected", C["accent"])])

        self._nb = ttk.Notebook(self)
        self._nb.pack(fill="both", expand=True, padx=0, pady=0)

        # Build each tab
        self._tab_live     = self._make_frame()
        self._tab_signals  = self._make_frame()
        self._tab_history  = self._make_frame()
        self._tab_jitai    = self._make_frame()
        self._tab_settings = self._make_frame()
        self._tab_export   = self._make_frame()

        self._nb.add(self._tab_live,     text="  [CAM]  LIVE DETECTION  ")
        self._nb.add(self._tab_signals,  text="  [SIG]  SIGNAL GRAPHS  ")
        self._nb.add(self._tab_history,  text="  [CHT]  STRESS HISTORY  ")
        self._nb.add(self._tab_jitai,    text="  [TIP]  JITAI  ")
        self._nb.add(self._tab_settings, text="  [CFG]  SETTINGS  ")
        self._nb.add(self._tab_export,   text="  [EXP]  EXPORT  ")

        self._build_live_tab()
        self._build_signals_tab()
        self._build_history_tab()
        self._build_jitai_tab()
        self._build_settings_tab()
        self._build_export_tab()

    def _make_frame(self) -> tk.Frame:
        f = tk.Frame(self._nb, bg=C["bg"])
        return f

    # ── TAB: LIVE ─────────────────────────────────────────────────────────────
    def _build_live_tab(self):
        p = self._tab_live

        # Left — video feed
        left = tk.Frame(p, bg=C["bg"])
        left.pack(side="left", fill="both", expand=True, padx=(12, 6), pady=12)

        self._panel_hdr(left, "WEBCAM FEED")
        self._video_label = tk.Label(left, bg="#000000",
                                     relief="flat", bd=0)
        self._video_label.pack(fill="both", expand=True)

        # Control bar under video
        ctrl = tk.Frame(left, bg=C["bg"], height=40)
        ctrl.pack(fill="x", pady=(6, 0))
        ctrl.pack_propagate(False)

        self._btn_start = self._btn(ctrl, "[ START ]", self._start_detection,
                                    fg=C["green"])
        self._btn_start.pack(side="left", padx=4)
        self._btn_stop = self._btn(ctrl, "[ STOP ]", self._stop_detection,
                                   fg=C["red"], state="disabled")
        self._btn_stop.pack(side="left", padx=4)

        tk.Label(ctrl, textvariable=self._countdown_var,
                 font=("Courier New", 13, "bold"), fg=C["accent2"], bg=C["bg"]).pack(side="left", padx=12)

        self._fps_var = tk.StringVar(value="FPS: —")
        tk.Label(ctrl, textvariable=self._fps_var,
                 font=FONT_SMALL, fg=C["text_dim"], bg=C["bg"]).pack(side="right", padx=8)

        # Right — readout panel
        right = tk.Frame(p, bg=C["bg"], width=300)
        right.pack(side="right", fill="y", padx=(6, 12), pady=12)
        right.pack_propagate(False)

        self._panel_hdr(right, "BRANCH READOUTS")

        # CNN card
        cnn_card = self._card(right)
        cnn_card.pack(fill="x", pady=(4, 3))
        tk.Label(cnn_card, text="[CNN]  CNN BRANCH (20%)",
                 font=FONT_SECTION, fg=C["accent"], bg=C["panel2"]).pack(anchor="w")
        self._cnn_emotion_var = tk.StringVar(value="Emotion:  —")
        self._cnn_conf_var    = tk.StringVar(value="Confidence:  —")
        self._cnn_score_var   = tk.StringVar(value="Stress score:  —")
        for v in (self._cnn_emotion_var, self._cnn_conf_var, self._cnn_score_var):
            tk.Label(cnn_card, textvariable=v,
                     font=FONT_LABEL, fg=C["text"], bg=C["panel2"],
                     anchor="w").pack(fill="x", pady=1)

        # Physio card
        phy_card = self._card(right)
        phy_card.pack(fill="x", pady=3)
        tk.Label(phy_card, text="[PHY]  PHYSIO BRANCH (80%)",
                 font=FONT_SECTION, fg=C["green"], bg=C["panel2"]).pack(anchor="w")
        self._sig_vars = {}
        sig_names = ["blink_rate", "eye_openness", "gaze_jitter",
                     "head_motion", "brow_tension", "jaw_tension"]
        sig_labels = {
            "blink_rate":   "Blink Rate   ×0.25",
            "eye_openness": "Eye Openness ×0.16",
            "gaze_jitter":  "Gaze Jitter  ×0.16",
            "head_motion":  "Head Motion  ×0.12",
            "brow_tension": "Brow Tension ×0.08",
            "jaw_tension":  "Jaw Tension  ×0.08",
        }
        for sig in sig_names:
            row = tk.Frame(phy_card, bg=C["panel2"])
            row.pack(fill="x", pady=1)
            v = tk.StringVar(value="0.000")
            self._sig_vars[sig] = v
            tk.Label(row, text=sig_labels[sig],
                     font=FONT_SMALL, fg=SIGNAL_COLORS[sig],
                     bg=C["panel2"], width=22, anchor="w").pack(side="left")
            tk.Label(row, textvariable=v,
                     font=FONT_SMALL, fg=C["text"], bg=C["panel2"]).pack(side="left")

        self._physio_score_var = tk.StringVar(value="Physio score:  —")
        tk.Label(phy_card, textvariable=self._physio_score_var,
                 font=FONT_LABEL, fg=C["green"], bg=C["panel2"],
                 anchor="w").pack(fill="x", pady=(4, 1))

        # Fuser card
        fuser_card = self._card(right, border_color=C["accent"])
        fuser_card.pack(fill="x", pady=3)
        tk.Label(fuser_card, text="[>=<]  SCORE FUSER",
                 font=FONT_SECTION, fg=C["accent"], bg=C["panel2"]).pack(anchor="w")
        self._fuser_var = tk.StringVar(value="0.20 × 0.000 + 0.80 × 0.000 = 0.000")
        tk.Label(fuser_card, textvariable=self._fuser_var,
                 font=FONT_SMALL, fg=C["text_dim"], bg=C["panel2"],
                 wraplength=260, justify="left").pack(anchor="w")
        self._fused_score_var = tk.StringVar(value="—  / 100")
        tk.Label(fuser_card, textvariable=self._fused_score_var,
                 font=FONT_BIG, fg=C["accent2"], bg=C["panel2"]).pack(anchor="w", pady=(4, 0))

        # Stress badge
        self._stress_frame = tk.Frame(right, bg=C["bg"], height=60)
        self._stress_frame.pack(fill="x", pady=(6, 3))
        self._stress_frame.pack_propagate(False)
        self._stress_badge_var = tk.StringVar(value="—")
        self._stress_badge = tk.Label(self._stress_frame,
                                      textvariable=self._stress_badge_var,
                                      font=("Courier", 18, "bold"),
                                      fg=C["text_dim"], bg=C["panel"],
                                      relief="flat")
        self._stress_badge.pack(fill="both", expand=True)

        # Mini suggestion
        self._mini_sug_var = tk.StringVar(value="")
        tk.Label(right, textvariable=self._mini_sug_var,
                 font=FONT_SMALL, fg=C["accent2"], bg=C["bg"],
                 wraplength=270, justify="left").pack(fill="x", pady=(4, 0))

    # ── TAB: SIGNAL GRAPHS ────────────────────────────────────────────────────
    def _build_signals_tab(self):
        p = self._tab_signals
        self._panel_hdr(p, "REAL-TIME PHYSIOLOGICAL SIGNALS  (last 150 frames)")

        if not MPL_AVAILABLE:
            tk.Label(p, text="matplotlib not installed.\npip install matplotlib",
                     font=FONT_LABEL, fg=C["red"], bg=C["bg"]).pack(expand=True)
            self._sig_canvas = None
            return

        fig = Figure(figsize=(12, 7), facecolor=C["bg"])
        fig.subplots_adjust(hspace=0.55, left=0.06, right=0.97,
                            top=0.94, bottom=0.06)

        self._sig_axes = {}
        self._sig_lines = {}
        sig_keys = list(SIGNAL_COLORS.keys())

        for i, key in enumerate(sig_keys):
            ax = fig.add_subplot(3, 2, i + 1)
            ax.set_facecolor(C["panel"])
            ax.tick_params(colors=C["text_dim"], labelsize=7)
            ax.spines[:].set_color(C["border"])
            for spine in ax.spines.values():
                spine.set_linewidth(0.5)
            ax.set_ylim(0, 1.05)
            ax.set_xlim(0, HISTORY_LEN)
            ax.set_title(key.replace("_", " ").title(),
                         color=SIGNAL_COLORS[key], fontsize=9,
                         fontfamily="monospace", pad=3)
            ax.set_yticks([0, 0.5, 1.0])
            ax.set_xticks([])
            ax.grid(True, color=C["border"], linewidth=0.4, alpha=0.6)

            line, = ax.plot([], [], color=SIGNAL_COLORS[key],
                            linewidth=1.4, alpha=0.9)
            self._sig_axes[key]  = ax
            self._sig_lines[key] = line

        canvas = FigureCanvasTkAgg(fig, master=p)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=4)
        self._sig_canvas = canvas
        self._sig_fig    = fig

    # ── TAB: HISTORY ──────────────────────────────────────────────────────────
    def _build_history_tab(self):
        p = self._tab_history
        self._panel_hdr(p, "SESSION STRESS HISTORY")

        if not MPL_AVAILABLE:
            tk.Label(p, text="matplotlib not installed.\npip install matplotlib",
                     font=FONT_LABEL, fg=C["red"], bg=C["bg"]).pack(expand=True)
            self._hist_canvas = None
            return

        fig = Figure(figsize=(12, 5), facecolor=C["bg"])
        fig.subplots_adjust(left=0.06, right=0.97, top=0.88, bottom=0.12)

        ax = fig.add_subplot(1, 1, 1)
        ax.set_facecolor(C["panel"])
        ax.tick_params(colors=C["text_dim"], labelsize=8)
        ax.spines[:].set_color(C["border"])
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
        ax.set_ylim(-0.1, 1.1)
        ax.set_yticks([0, 0.5, 1.0])
        ax.set_yticklabels(["0", "0.5", "1.0"], color=C["text_dim"])
        ax.set_title("Fused Stress Score  (0=Low  0.5=Medium  1=High)",
                     color=C["text_dim"], fontsize=9, fontfamily="monospace")
        ax.grid(True, color=C["border"], linewidth=0.4, alpha=0.5)

        # Threshold bands
        ax.axhspan(0,    0.33, color="#00e67618", zorder=0)
        ax.axhspan(0.33, 0.66, color="#ffab4018", zorder=0)
        ax.axhspan(0.66, 1.0,  color="#ff444418", zorder=0)

        self._hist_line, = ax.plot([], [], color=C["accent"],
                                   linewidth=1.6, alpha=0.9)
        self._hist_fill  = ax.fill_between([], [], alpha=0.15,
                                            color=C["accent"])
        self._hist_ax = ax

        canvas = FigureCanvasTkAgg(fig, master=p)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=4)
        self._hist_canvas = canvas
        self._hist_fig    = fig

        # Stats row
        stats = tk.Frame(p, bg=C["bg"])
        stats.pack(fill="x", padx=12, pady=(0, 8))
        self._stat_low  = self._stat_box(stats, "LOW",    "0", C["green"])
        self._stat_med  = self._stat_box(stats, "MEDIUM", "0", C["amber"])
        self._stat_high = self._stat_box(stats, "HIGH",   "0", C["red"])
        self._stat_avg  = self._stat_box(stats, "AVG SCORE", "0.00", C["accent"])
        for w in (self._stat_low, self._stat_med, self._stat_high, self._stat_avg):
            w.pack(side="left", padx=6)

    # ── TAB: JITAI ────────────────────────────────────────────────────────────
    def _build_jitai_tab(self):
        p = self._tab_jitai
        self._panel_hdr(p, "JUST-IN-TIME ADAPTIVE INTERVENTIONS (JITAI)")

        # Detected stress level banner
        banner = tk.Frame(p, bg=C["panel2"], bd=0, relief="flat")
        banner.pack(fill="x", padx=12, pady=(4, 8))
        tk.Label(banner, text="DETECTED STRESS LEVEL",
                 font=FONT_SMALL, fg=C["text_dim"], bg=C["panel2"]).pack(anchor="w", padx=10, pady=(6, 0))
        self._jitai_level_var = tk.StringVar(value="-- Run a 5s analysis first --")
        self._jitai_level_lbl = tk.Label(banner, textvariable=self._jitai_level_var,
                 font=("Courier New", 16, "bold"),
                 fg=C["accent2"], bg=C["panel2"],
                 wraplength=900, justify="left")
        self._jitai_level_lbl.pack(anchor="w", padx=10, pady=(2, 10))

        tk.Label(p, text="RECOMMENDED INTERVENTIONS",
                 font=FONT_SECTION, fg=C["text_dim"], bg=C["bg"]).pack(anchor="w", padx=14, pady=(4, 2))

        self._jitai_tips_frame = tk.Frame(p, bg=C["bg"])
        self._jitai_tips_frame.pack(fill="both", expand=True, padx=12, pady=4)

        self._jitai_placeholder = tk.Label(self._jitai_tips_frame,
                 text="Tips will appear here after a 5-second analysis.",
                 font=FONT_LABEL, fg=C["text_dim"], bg=C["bg"])
        self._jitai_placeholder.pack(anchor="w", pady=20)

        # Keep jitai_var for backward compat
        self._jitai_var = self._jitai_level_var



    # ── TAB: SETTINGS ─────────────────────────────────────────────────────────
    def _build_settings_tab(self):
        p = self._tab_settings
        self._panel_hdr(p, "SETTINGS")

        content = tk.Frame(p, bg=C["bg"])
        content.pack(fill="both", expand=True, padx=20, pady=10)

        # Camera
        self._setting_section(content, "CAMERA")

        # WSL / usbipd hint
        import platform
        if "microsoft" in platform.uname().release.lower() or "wsl" in platform.uname().release.lower():
            tk.Label(content,
                     text="WSL detected -- attach webcam with:  usbipd attach --wsl --busid <ID>",
                     font=FONT_SMALL, fg=C["amber"], bg=C["bg"],
                     wraplength=700, justify="left").pack(anchor="w", pady=(0, 4))

        cam_row = tk.Frame(content, bg=C["bg"])
        cam_row.pack(fill="x", pady=3)
        tk.Label(cam_row, text="Camera ID:  ", font=FONT_LABEL,
                 fg=C["text"], bg=C["bg"]).pack(side="left")
        self._cam_var = tk.IntVar(value=self._settings["camera_id"])
        tk.Spinbox(cam_row, from_=0, to=10, textvariable=self._cam_var,
                   width=5, font=FONT_LABEL,
                   bg=C["panel2"], fg=C["text"], insertbackground=C["accent"],
                   buttonbackground=C["panel"],
                   relief="flat").pack(side="left")

        self._btn(cam_row, "Scan for cameras", self._scan_cameras,
                  fg=C["accent"]).pack(side="left", padx=10)
        self._cam_scan_var = tk.StringVar(value="")
        tk.Label(cam_row, textvariable=self._cam_scan_var,
                 font=FONT_SMALL, fg=C["green"], bg=C["bg"]).pack(side="left")

        # Detection
        self._setting_section(content, "DETECTION")

        nb_row = tk.Frame(content, bg=C["bg"])
        nb_row.pack(fill="x", pady=3)
        tk.Label(nb_row, text="Haar minNeighbors (higher = fewer false positives):  ",
                 font=FONT_LABEL, fg=C["text"], bg=C["bg"]).pack(side="left")
        self._neighbors_var = tk.IntVar(value=self._settings["min_neighbors"])
        tk.Spinbox(nb_row, from_=2, to=10, textvariable=self._neighbors_var,
                   width=4, font=FONT_LABEL,
                   bg=C["panel2"], fg=C["text"],
                   buttonbackground=C["panel"], relief="flat").pack(side="left")

        # Overlays
        self._setting_section(content, "OVERLAYS")
        self._overlay_var = tk.BooleanVar(value=self._settings["show_overlay"])
        self._debug_var   = tk.BooleanVar(value=self._settings["show_debug"])
        self._check(content, "Show CV2 overlays on video",   self._overlay_var)
        self._check(content, "Show physio signal bars on video", self._debug_var)

        # Fusion weights
        self._setting_section(content, "FUSION WEIGHTS")
        cw_row = tk.Frame(content, bg=C["bg"])
        cw_row.pack(fill="x", pady=3)
        tk.Label(cw_row, text=f"CNN weight:    {CNN_WEIGHT*100:.0f}%  |  "
                              f"Physio weight: {PHYSIO_WEIGHT*100:.0f}%",
                 font=FONT_LABEL, fg=C["text_dim"], bg=C["bg"]).pack(side="left")
        tk.Label(content,
                 text="(Weights are fixed per architecture — retrain to change.)",
                 font=FONT_SMALL, fg=C["text_dim"], bg=C["bg"]).pack(anchor="w")

        # Apply button
        tk.Frame(content, bg=C["bg"], height=12).pack()
        self._btn(content, "[ APPLY SETTINGS ]", self._apply_settings,
                  fg=C["green"]).pack(anchor="w")

        self._settings_status = tk.StringVar(value="")
        tk.Label(content, textvariable=self._settings_status,
                 font=FONT_SMALL, fg=C["green"], bg=C["bg"]).pack(anchor="w")

    # ── TAB: EXPORT ───────────────────────────────────────────────────────────
    def _build_export_tab(self):
        p = self._tab_export
        self._panel_hdr(p, "SESSION REPORT & EXPORT")

        content = tk.Frame(p, bg=C["bg"])
        content.pack(fill="both", expand=True, padx=20, pady=10)

        # Summary
        self._setting_section(content, "SESSION SUMMARY")
        self._export_summary_var = tk.StringVar(value="No session data yet.")
        tk.Label(content, textvariable=self._export_summary_var,
                 font=FONT_LABEL, fg=C["text"], bg=C["bg"],
                 justify="left").pack(anchor="w", pady=(4, 12))

        # Export options
        self._setting_section(content, "EXPORT OPTIONS")

        btn_row = tk.Frame(content, bg=C["bg"])
        btn_row.pack(fill="x", pady=6)

        self._btn(btn_row, "[ Export CSV (raw log) ]",
                  self._export_csv, fg=C["accent"]).pack(side="left", padx=6)
        self._btn(btn_row, "[ Export TXT (summary) ]",
                  self._export_txt, fg=C["accent"]).pack(side="left", padx=6)
        self._btn(btn_row, "[ Clear Session Data ]",
                  self._clear_session, fg=C["red"]).pack(side="left", padx=6)

        self._export_status_var = tk.StringVar(value="")
        tk.Label(content, textvariable=self._export_status_var,
                 font=FONT_SMALL, fg=C["green"], bg=C["bg"]).pack(anchor="w", pady=4)

        # Preview log table
        self._setting_section(content, "RECENT LOG  (last 10 entries)")
        self._log_text = tk.Text(content, height=12,
                                 bg=C["panel"], fg=C["text_dim"],
                                 font=("Courier New", 9), relief="flat",
                                 state="disabled", wrap="none")
        self._log_text.pack(fill="x", pady=4)

    # ── UI HELPERS ────────────────────────────────────────────────────────────

    def _panel_hdr(self, parent, text: str):
        f = tk.Frame(parent, bg=C["bg"])
        f.pack(fill="x", padx=12, pady=(8, 4))
        tk.Label(f, text=text, font=FONT_SECTION,
                 fg=C["text_dim"], bg=C["bg"]).pack(side="left")
        tk.Frame(f, bg=C["border"], height=1).pack(side="left", fill="x",
                                                    expand=True, padx=(8, 0))

    def _card(self, parent, border_color=None) -> tk.Frame:
        bc = border_color or C["border"]
        outer = tk.Frame(parent, bg=bc, bd=1)
        inner = tk.Frame(outer, bg=C["panel2"], padx=8, pady=6)
        inner.pack(fill="both", expand=True, padx=1, pady=1)
        return inner

    def _btn(self, parent, text, command, fg=None, state="normal") -> tk.Button:
        return tk.Button(parent, text=text, command=command,
                         font=FONT_LABEL,
                         fg=fg or C["text"],
                         bg=C["panel2"],
                         activeforeground=C["text_bright"],
                         activebackground=C["panel"],
                         relief="flat", bd=0,
                         padx=10, pady=4,
                         cursor="hand2",
                         state=state)

    def _check(self, parent, text, var):
        tk.Checkbutton(parent, text=text, variable=var,
                       font=FONT_LABEL, fg=C["text"], bg=C["bg"],
                       selectcolor=C["panel2"],
                       activeforeground=C["text"],
                       activebackground=C["bg"]).pack(anchor="w", pady=2)

    def _setting_section(self, parent, text: str):
        tk.Label(parent, text=text,
                 font=FONT_SECTION, fg=C["accent"], bg=C["bg"]).pack(
            anchor="w", pady=(12, 2))

    def _stat_box(self, parent, label: str, value: str, color: str) -> tk.Frame:
        f = tk.Frame(parent, bg=C["panel2"], padx=12, pady=6)
        tk.Label(f, text=label, font=FONT_SMALL,
                 fg=C["text_dim"], bg=C["panel2"]).pack()
        v = tk.StringVar(value=value)
        tk.Label(f, textvariable=v, font=FONT_MED,
                 fg=color, bg=C["panel2"]).pack()
        f._val_var = v
        return f

    # ── MODEL / PHYSIO LOADING ────────────────────────────────────────────────

    def _load_model_and_physio(self):
        """Load CNN model and init PhysioBranch in background thread."""
        def _load():
            self._update_status("[..] LOADING MODEL", C["amber"])
            model_path = PROJECT_ROOT / "models" / "stress_emotion_model.h5"
            if model_path.exists():
                try:
                    import tensorflow as tf
                    from tensorflow import keras
                    self._cnn_model = keras.models.load_model(str(model_path))
                    self._update_status("[OK] MODEL LOADED", C["green"])
                except Exception as e:
                    self._update_status(f"[!!] MODEL ERROR: {e}", C["red"])
                    return
            else:
                self._update_status("[!!] MODEL NOT FOUND", C["red"])
                return

            self._update_status("[..] INIT MEDIAPIPE", C["amber"])
            try:
                self._physio = PhysioBranch()
                if self._physio.available:
                    self._update_status("[OK] READY", C["green"])
                else:
                    self._update_status("[OK] READY (no MediaPipe)", C["amber"])
            except Exception as e:
                self._update_status(f"[!!] PHYSIO ERROR: {e}", C["red"])

        threading.Thread(target=_load, daemon=True).start()

    def _update_status(self, text: str, color: str):
        def _do():
            self._status_var.set(text)
            self._status_lbl.configure(fg=color)
        self.after(0, _do)

    # ── DETECTION CONTROL ─────────────────────────────────────────────────────

    def _start_detection(self):
        if self._cnn_model is None:
            messagebox.showwarning("Not ready",
                                   "CNN model not loaded yet. Please wait.")
            return

        # First click: start the video feed
        if not self._running:
            self._running = True
            self._btn_start.configure(text="[ ANALYSE 5s ]")
            self._btn_stop.configure(state="normal")
            self._update_status("[**] LIVE", C["accent"])
            self._thread = DetectionThread(
                self._result_queue, self._settings,
                self._cnn_model, self._physio, self._jitai
            )
            self._thread.start()
            self._poll_results()
            return

        # Subsequent clicks: trigger a 5s analysis window (video keeps running)
        if self._capture_mode:
            return  # already analysing
        self._capture_mode   = True
        self._capture_frames = []
        self._capture_end    = time.time() + 5.0
        self._countdown_var.set("Analysing... 5.0s")
        self._update_status("[**] ANALYSING", C["amber"])

    def _stop_detection(self):
        if not self._running:
            return
        self._running      = False
        self._capture_mode = False
        self._countdown_var.set("")
        if self._thread:
            self._thread.stop()
            self._thread = None
        self._btn_start.configure(text="[ START ]", state="normal")
        self._btn_stop.configure(state="disabled")
        self._update_status("[--] OFFLINE", C["text_dim"])
        self._video_label.configure(image="", bg="#000000")
        self._video_label._photo = None

    def _poll_results(self):
        """Called every ~33 ms — video runs forever, 5s analysis window is optional."""
        if not self._running:
            return

        # Always update video frame
        if self._thread:
            frame = self._thread.get_frame()
            if frame is not None:
                self._display_frame(frame)

        # Countdown ticker — independent of face detection
        if self._capture_mode:
            remaining = self._capture_end - time.time()
            if remaining > 0:
                self._countdown_var.set(f"Analysing... {remaining:.1f}s")
            else:
                # Time's up — show result, video keeps running
                self._capture_mode = False
                self._countdown_var.set("Done! See JITAI tab")
                self._finalise_prediction()
                self._update_status("[**] LIVE", C["accent"])
                self.after(3000, lambda: self._countdown_var.set(""))

        # Drain latest result from queue
        result = None
        while True:
            try:
                result = self._result_queue.get_nowait()
            except queue.Empty:
                break

        if result:
            if "error" in result:
                messagebox.showerror("Camera error", result["error"])
                self._stop_detection()
                return

            # Collect frames during capture window
            if self._capture_mode:
                self._capture_frames.append(result)

            # Always update live readouts on side panel
            self._apply_result(result)

        self.after(33, self._poll_results)

    def _finalise_prediction(self):
        """Average all captured frames → final stress level → show all JITAI tips."""
        if not self._capture_frames:
            return

        # Average fused scores over 5s window
        fused_scores = [r["fused_score"] for r in self._capture_frames]
        avg_fused    = float(np.mean(fused_scores))
        from data_loader import fused_score_to_label
        final_label, _ = fused_score_to_label(avg_fused)

        # Most common emotion
        emotions = [r["emotion"] for r in self._capture_frames if r["emotion"] != "—"]
        dominant_emotion = max(set(emotions), key=emotions.count) if emotions else "—"

        # Update stress badge with final result
        color_map = {"High": C["red"], "Medium": C["amber"], "Low": C["green"]}
        badge_map = {"High": "[!!!]  HIGH STRESS", "Medium": "[!]  MEDIUM STRESS", "Low": "[OK]  LOW STRESS"}
        self._stress_badge_var.set(badge_map.get(final_label, "—"))
        self._stress_badge.configure(fg=color_map.get(final_label, C["text_dim"]), bg=C["panel"])
        self._fused_score_var.set(f"{avg_fused*100:.0f}  / 100  [FINAL]")
        self._update_status(f"[OK] RESULT: {final_label.upper()}", color_map.get(final_label, C["green"]))

        # Populate JITAI tab with ALL tips for this stress level
        self._show_all_jitai_tips(final_label, avg_fused, dominant_emotion)

        # Switch to JITAI tab automatically
        self._nb.select(self._tab_jitai)

    def _show_all_jitai_tips(self, level: str, avg_score: float, emotion: str):
        """Clear and repopulate JITAI tips frame with all tips for detected level."""
        color_map = {"High": C["red"], "Medium": C["amber"], "Low": C["green"]}
        color = color_map.get(level, C["text_dim"])

        # Update level label
        self._jitai_level_var.set(
            f"{level.upper()} STRESS  |  Score: {avg_score*100:.0f}/100  |  Dominant emotion: {emotion}"
        )
        self._jitai_level_lbl.configure(fg=color)

        # Clear existing tips
        for widget in self._jitai_tips_frame.winfo_children():
            widget.destroy()

        # Show ALL tips for this level
        tips = JITAI_SUGGESTIONS.get(level, [])
        for i, tip in enumerate(tips, 1):
            row = tk.Frame(self._jitai_tips_frame, bg=C["panel2"], pady=6, padx=10)
            row.pack(fill="x", pady=4)
            tk.Label(row, text=f"{i}.", font=("Courier New", 13, "bold"),
                     fg=color, bg=C["panel2"], width=3).pack(side="left", anchor="n")
            tk.Label(row, text=tip, font=("Courier New", 12),
                     fg=C["text"], bg=C["panel2"],
                     wraplength=800, justify="left", anchor="w").pack(side="left", fill="x", expand=True)

        # Also update mini suggestion on live tab
        if tips:
            self._mini_sug_var.set(f"[RESULT] {level} stress — see JITAI tab for all tips")

    def _display_frame(self, frame_bgr: np.ndarray):
        """Resize frame to fit label and display via PIL/ImageTk."""
        try:
            from PIL import Image, ImageTk
        except ImportError:
            return   # Pillow required for display

        lw = self._video_label.winfo_width()
        lh = self._video_label.winfo_height()
        if lw < 10 or lh < 10:
            lw, lh = 640, 480

        fh, fw = frame_bgr.shape[:2]
        scale  = min(lw / fw, lh / fh)
        nw, nh = int(fw * scale), int(fh * scale)

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb).resize((nw, nh), Image.BILINEAR)
        photo = ImageTk.PhotoImage(img)
        self._video_label.configure(image=photo)
        self._video_label._photo = photo   # prevent GC

    def _apply_result(self, r: dict):
        """Update all UI widgets from a detection result dict."""
        emotion       = r["emotion"]
        stress        = r["stress"]
        conf          = r["confidence"]
        es            = r["emotion_score"]
        ps            = r["physio_score"]
        fs            = r["fused_score"]
        sig           = r.get("physio_signals", {})
        fps           = r.get("fps", 0.0)

        # ── Rolling history ───────────────────────────────────────────────────
        s_num = {"High": 1.0, "Medium": 0.5, "Low": 0.0}.get(stress, 0.0)
        self._hist_fused.append(fs)
        self._hist_stress.append(s_num)
        self._hist_emotion.append(emotion)
        for k in SIGNAL_COLORS:
            self._hist_signals[k].append(sig.get(k, 0.0))

        # ── Session log ───────────────────────────────────────────────────────
        self._session_log.append(r)

        # ── Live tab — show emotion + physio always ───────────────────────────
        self._cnn_emotion_var.set(f"Emotion:      {emotion}")
        self._cnn_conf_var.set(   f"Confidence:   {conf*100:.1f}%")
        self._cnn_score_var.set(  f"Stress score: {es:.3f}")

        for k, v in self._sig_vars.items():
            v.set(f"{sig.get(k, 0.0):.3f}")
        self._physio_score_var.set(f"Physio score:  {ps:.3f}")

        self._fuser_var.set(
            f"0.20 x {es:.3f}  +  0.80 x {ps:.3f}  =  {fs:.3f}"
        )

        # ── Stress badge — only update during capture or after final result ───
        # (stays blank / shows last final result during normal live view)
        if self._capture_mode:
            self._fused_score_var.set(f"{fs*100:.0f}  / 100")

        self._fps_var.set(f"FPS: {fps:.1f}")

        # ── Signal graphs (every 3rd result to avoid over-drawing) ────────────
        if len(self._hist_fused) % 3 == 0:
            self._refresh_signal_graphs()
            self._refresh_history_chart()
            self._refresh_stats()
            self._refresh_export_preview()

    def _refresh_signal_graphs(self):
        if not MPL_AVAILABLE or self._sig_canvas is None:
            return
        for key, line in self._sig_lines.items():
            data = list(self._hist_signals[key])
            xs   = list(range(len(data)))
            line.set_data(xs, data)
            self._sig_axes[key].set_xlim(0, max(HISTORY_LEN, len(data)))
        self._sig_canvas.draw_idle()

    def _refresh_history_chart(self):
        if not MPL_AVAILABLE or self._hist_canvas is None:
            return
        data = list(self._hist_fused)
        xs   = list(range(len(data)))
        self._hist_line.set_data(xs, data)
        self._hist_ax.set_xlim(0, max(HISTORY_LEN, len(data)))

        # Redraw fill_between
        for coll in self._hist_ax.collections:
            coll.remove()
        if len(xs) > 1:
            self._hist_ax.fill_between(xs, data, alpha=0.12, color=C["accent"])
        # Re-draw threshold bands
        self._hist_ax.axhspan(0,    0.33, color="#00e67618", zorder=0)
        self._hist_ax.axhspan(0.33, 0.66, color="#ffab4018", zorder=0)
        self._hist_ax.axhspan(0.66, 1.0,  color="#ff444418", zorder=0)

        self._hist_canvas.draw_idle()

    def _refresh_stats(self):
        stresses = [r["stress"] for r in self._session_log]
        n = len(stresses) or 1
        low    = stresses.count("Low")
        med    = stresses.count("Medium")
        high   = stresses.count("High")
        avg    = np.mean(list(self._hist_fused)) if self._hist_fused else 0.0

        self._stat_low._val_var.set(str(low))
        self._stat_med._val_var.set(str(med))
        self._stat_high._val_var.set(str(high))
        self._stat_avg._val_var.set(f"{avg:.2f}")

        summary = (
            f"Total frames:  {len(self._session_log)}\n"
            f"Low stress:    {low}  ({low/n*100:.1f}%)\n"
            f"Medium stress: {med}  ({med/n*100:.1f}%)\n"
            f"High stress:   {high}  ({high/n*100:.1f}%)\n"
            f"Avg fused score: {avg:.3f}"
        )
        self._export_summary_var.set(summary)

    def _refresh_export_preview(self):
        last = self._session_log[-10:]
        self._log_text.configure(state="normal")
        self._log_text.delete("1.0", "end")
        header = f"{'Time':<12}  {'Emotion':<10}  {'Stress':<8}  {'Fused':>6}  {'Physio':>6}\n"
        self._log_text.insert("end", header)
        self._log_text.insert("end", "─" * 54 + "\n")
        for r in last:
            t   = r["timestamp"][11:19]
            row = (f"{t:<12}  {r['emotion']:<10}  {r['stress']:<8}  "
                   f"{r['fused_score']:>6.3f}  {r['physio_score']:>6.3f}\n")
            self._log_text.insert("end", row)
        self._log_text.configure(state="disabled")

    # ── SETTINGS ──────────────────────────────────────────────────────────────

    def _scan_cameras(self):
        """Scan indices 0-5 for available cameras and report found ones."""
        self._cam_scan_var.set("Scanning...")
        self.update_idletasks()
        found = []
        for i in range(6):
            for backend in [cv2.CAP_DSHOW, cv2.CAP_ANY]:
                try:
                    cap = cv2.VideoCapture(i, backend)
                    if cap.isOpened():
                        ret, _ = cap.read()
                        cap.release()
                        if ret:
                            found.append(i)
                            break
                    else:
                        cap.release()
                except Exception:
                    pass
        if found:
            self._cam_scan_var.set(f"Found: {found}  -- set ID above")
            self._cam_var.set(found[0])
        else:
            self._cam_scan_var.set("No cameras found")

    def _apply_settings(self):
        was_running = self._running
        if was_running:
            self._stop_detection()

        self._settings["camera_id"]     = self._cam_var.get()
        self._settings["min_neighbors"] = self._neighbors_var.get()
        self._settings["show_overlay"]  = self._overlay_var.get()
        self._settings["show_debug"]    = self._debug_var.get()

        self._settings_status.set("Settings applied.")
        self.after(2000, lambda: self._settings_status.set(""))

        if was_running:
            self._start_detection()

    # ── JITAI MANUAL TRIGGER ──────────────────────────────────────────────────

    def _trigger_suggestion(self, level: str):
        self._show_all_jitai_tips(level, {"High": 0.8, "Medium": 0.5, "Low": 0.2}[level], "—")
        self._nb.select(self._tab_jitai)

    # ── EXPORT ────────────────────────────────────────────────────────────────

    def _export_csv(self):
        if not self._session_log:
            messagebox.showinfo("No data", "Run a detection session first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            initialfile=f"stress_session_{datetime.now():%Y%m%d_%H%M%S}.csv"
        )
        if not path:
            return
        keys = ["timestamp", "emotion", "stress", "confidence",
                "emotion_score", "physio_score", "fused_score"]
        sig_keys = list(SIGNAL_COLORS.keys())
        try:
            with open(path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=keys + sig_keys,
                                        extrasaction="ignore")
                writer.writeheader()
                for r in self._session_log:
                    row = {k: r.get(k, "") for k in keys}
                    sig = r.get("physio_signals", {})
                    for sk in sig_keys:
                        row[sk] = sig.get(sk, 0.0)
                    writer.writerow(row)
            self._export_status_var.set(f"[v]  CSV saved → {path}")
            self.after(4000, lambda: self._export_status_var.set(""))
        except Exception as e:
            messagebox.showerror("Export error", str(e))

    def _export_txt(self):
        if not self._session_log:
            messagebox.showinfo("No data", "Run a detection session first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt")],
            initialfile=f"stress_report_{datetime.now():%Y%m%d_%H%M%S}.txt"
        )
        if not path:
            return

        stresses = [r["stress"] for r in self._session_log]
        n = len(stresses) or 1
        low  = stresses.count("Low")
        med  = stresses.count("Medium")
        high = stresses.count("High")
        fused_vals = [r["fused_score"] for r in self._session_log]
        avg  = np.mean(fused_vals) if fused_vals else 0.0
        peak = max(fused_vals, default=0.0)

        try:
            with open(path, "w") as f:
                f.write("=" * 60 + "\n")
                f.write("  DUAL-BRANCH STRESS DETECTION — SESSION REPORT\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"Generated : {datetime.now():%Y-%m-%d %H:%M:%S}\n")
                f.write(f"Frames    : {n}\n")
                f.write(f"Duration  : ~{n/30:.0f} seconds @ 30 fps\n\n")

                f.write("STRESS DISTRIBUTION\n")
                f.write("-" * 30 + "\n")
                f.write(f"  Low    : {low:>5}  ({low/n*100:>5.1f}%)\n")
                f.write(f"  Medium : {med:>5}  ({med/n*100:>5.1f}%)\n")
                f.write(f"  High   : {high:>5}  ({high/n*100:>5.1f}%)\n\n")

                f.write("FUSED SCORE STATISTICS\n")
                f.write("-" * 30 + "\n")
                f.write(f"  Average : {avg:.3f}  ({avg*100:.0f}/100)\n")
                f.write(f"  Peak    : {peak:.3f}  ({peak*100:.0f}/100)\n")
                f.write(f"  Min     : {min(fused_vals):.3f}\n\n")

                f.write("ARCHITECTURE\n")
                f.write("-" * 30 + "\n")
                f.write(f"  CNN Branch weight   : {CNN_WEIGHT*100:.0f}%\n")
                f.write(f"  Physio Branch weight: {PHYSIO_WEIGHT*100:.0f}%\n")
                f.write(f"  Stress thresholds   : Low 0-33 | Medium 34-66 | High 67-100\n\n")

                f.write("LAST JITAI SUGGESTION\n")
                f.write("-" * 30 + "\n")
                f.write(f"  {self._current_suggestion or 'None'}\n\n")

                f.write("=" * 60 + "\n")

            self._export_status_var.set(f"[v]  Report saved → {path}")
            self.after(4000, lambda: self._export_status_var.set(""))
        except Exception as e:
            messagebox.showerror("Export error", str(e))

    def _clear_session(self):
        if messagebox.askyesno("Clear session",
                                "Clear all session data and history?"):
            self._session_log.clear()
            self._hist_fused.clear()
            self._hist_stress.clear()
            for k in self._hist_signals:
                self._hist_signals[k].clear()
            self._hist_emotion.clear()
            self._export_summary_var.set("No session data yet.")
            self._export_status_var.set("Session cleared.")
            self.after(2000, lambda: self._export_status_var.set(""))

    # ── WINDOW CLOSE ─────────────────────────────────────────────────────────

    def _on_close(self):
        self._stop_detection()
        if self._physio:
            self._physio.close()
        self.destroy()


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app = StressDetectionApp()
    app.mainloop()
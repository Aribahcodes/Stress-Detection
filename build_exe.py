import subprocess, sys, os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
APP_SCRIPT   = PROJECT_ROOT / "app" / "gui.py"

import cv2
cascade_dir = Path(cv2.data.haarcascades)

datas = [
    f"{PROJECT_ROOT / 'src'}{os.pathsep}src",
    f"{cascade_dir}{os.pathsep}cv2/data",
]

try:
    import mediapipe as mp
    datas.append(f"{Path(mp.__file__).parent}{os.pathsep}mediapipe")
except ImportError:
    pass

cmd = [
    sys.executable, "-m", "PyInstaller",
    "--noconfirm", "--clean", "--onedir", "--windowed",
    "--name", "StressDetection",
    "--distpath", str(PROJECT_ROOT / "dist"),
    "--workpath", str(PROJECT_ROOT / "build"),
    "--hidden-import", "tensorflow",
    "--hidden-import", "keras",
    "--hidden-import", "mediapipe",
    "--hidden-import", "mediapipe.tasks",
    "--hidden-import", "mediapipe.tasks.python",
    "--hidden-import", "mediapipe.tasks.python.vision",
    "--hidden-import", "cv2",
    "--hidden-import", "PIL",
    "--hidden-import", "PIL.ImageTk",
    "--hidden-import", "sklearn",
    "--hidden-import", "matplotlib.backends.backend_tkagg",
]
for d in datas:
    cmd += ["--add-data", d]
cmd.append(str(APP_SCRIPT))

print("Building... this takes 3-5 minutes")
result = subprocess.run(cmd)
if result.returncode == 0:
    out = PROJECT_ROOT / "dist" / "StressDetection"
    print(f"\nSUCCESS -> {out}")
    print("Copy models/stress_emotion_model.keras into that folder too.")
else:
    print("\nBUILD FAILED — check errors above")

# Dual-Branch Stress Detection System

A real-time stress detection system I built that combines two approaches: a CNN trained on FER2013 for facial emotion recognition, and MediaPipe for tracking physiological signals like blink rate, gaze, and head movement through a webcam. The two are fused together to give a more reliable stress estimate than either one alone.

## Why two branches?

Facial expression alone isn't a great stress indicator — someone can look neutral and still be stressed, or make a face that looks "sad" for reasons that have nothing to do with stress. So instead of relying only on the CNN's emotion output, I added a second branch that tracks physiological cues (blinking more, eyes narrowing, jitter in gaze, tense jaw/brow, etc.) using MediaPipe's face landmarks. The physio branch ends up carrying most of the weight (80%) in the final score, with the CNN's emotion output contributing the remaining 20%.

```
Webcam Frame
     │
     ├── CNN Branch (20%) — FER2013 model → 7 emotions → stress valence
     │
     └── Physio Branch (80%) — MediaPipe landmarks → 6 signals
             (blink rate, eye openness, gaze jitter,
              head motion, brow tension, jaw tension)

              ↓ combined into one score ↓
        Low (0–33) | Medium (34–66) | High (67–100)
```

## Project structure

```
├── app/
│   ├── streamlit_app.py     - web app (Streamlit)
│   └── gui.py                - desktop app (Tkinter)
├── src/
│   ├── data_loader.py        - loads FER2013 images
│   ├── preprocess.py         - preprocessing/augmentation
│   ├── model.py               - CNN architecture
│   ├── train.py               - training script
│   ├── physio_branch.py       - MediaPipe signal extraction
│   └── realtime.py            - live webcam detection (OpenCV)
├── notebooks/                 - same pipeline, but step-by-step in Jupyter
├── data/                       - FER2013 dataset (you download this, not included)
├── models/                     - trained model goes here (not included)
├── build_exe.py                - packages the desktop app into an .exe
├── requirements.txt
└── .gitignore
```

Data and trained models aren't included in this repo (see `.gitignore`) — they're too large to commit. Instructions for getting the dataset are below.

## Setup

1. **Create an environment** (I used conda, but venv works too):
```bash
conda create -n stress_detect python=3.10 -y
conda activate stress_detect
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Get the dataset.** Download FER2013 from [Kaggle](https://www.kaggle.com/datasets/msambare/fer2013) and place it like this:
```
data/raw/train/angry/, disgust/, fear/, happy/, neutral/, sad/, surprise/
data/raw/test/ (same folders)
```

4. **(Optional) GPU setup.** If you have an NVIDIA GPU, install CUDA 12.3 and check it's detected:
```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```
CPU works fine too, just slower for training.

## Running it

**Web app:**
```bash
streamlit run app/streamlit_app.py
```
Opens at `localhost:8501`. Load the model + physio branch from the sidebar, then use the Live Detection tab.

**Desktop app:**
```bash
python app/gui.py
```

**Live webcam (no UI, just OpenCV window):**
```bash
python src/realtime.py
```
Press `Q` to quit, `S` for a stress-relief suggestion, `D` to toggle the debug panel.

**Or just go through the notebooks** in order (`01` through `05`) if you want to see the whole pipeline step by step.

## Training your own model

If you don't have a trained model yet:
```bash
python src/train.py
```
This trains for up to 50 epochs (with early stopping), saves the best version to `models/stress_emotion_model.h5`, and logs training history to `outputs/training_history.json`.

Rough training time: ~30s/epoch on a decent GPU, 20-60 min/epoch on CPU only.

## Results so far

| Metric | Value |
|---|---|
| CNN emotion accuracy (7-class) | ~65% |
| Fused stress accuracy (3-class) | ~78% |
| Real-time FPS (GPU) | 25–30 |

## Common issues

- **`ModuleNotFoundError: mediapipe`** → `pip install mediapipe`
- **`data/raw/train` not found** → you skipped the dataset download step
- **Model file not found** → run `train.py` first, or download a pretrained one
- **Webcam won't open** → close other apps using the camera, or try `camera_id=1` in `realtime.py`
- **CUDA out of memory** → lower `BATCH_SIZE` in `src/preprocess.py`

## Building an .exe (Windows)

```bash
pip install pyinstaller
python build_exe.py
```
Then copy `models/stress_emotion_model.h5` into the output folder — it's not bundled automatically.

---

Built as a portfolio project — feedback welcome.

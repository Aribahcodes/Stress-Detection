import os
import numpy as np
from pathlib import Path
from PIL import Image

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW     = PROJECT_ROOT / "data" / "raw"
DATA_PROC    = PROJECT_ROOT / "data" / "processed"
DATA_PROC.mkdir(parents=True, exist_ok=True)

# ── Constants ──────────────────────────────────────────────────────────────────
IMG_SIZE    = 48        # FER2013 images are 48×48 pixels
NUM_CLASSES = 7

# Folder name → class index (must match subfolder names exactly)
LABEL_MAP = {
    "angry":    0,
    "disgust":  1,
    "fear":     2,
    "happy":    3,
    "sad":      4,
    "surprise": 5,
    "neutral":  6,
}

EMOTION_LABELS = {v: k.capitalize() for k, v in LABEL_MAP.items()}
EMOTION_LABELS[0] = "Angry"
EMOTION_LABELS[3] = "Happy"
EMOTION_LABELS[6] = "Neutral"

# ── CNN Branch: Emotion → stress valence (0.0–1.0) ────────────────────────────
# Architecture diagram: "Angry/Fear/Disgust → 1.0"
EMOTION_VALENCE = {
    "Angry":    1.0,   # High stress → maps to score 1.0
    "Disgust":  1.0,
    "Fear":     1.0,
    "Sad":      0.5,   # Medium stress → maps to score 0.5
    "Surprise": 0.5,
    "Neutral":  0.0,   # Low stress → maps to score 0.0
    "Happy":    0.0,
}

# Legacy stress map kept for UI badges (High / Medium / Low labels)
STRESS_MAP = {
    "Angry":    ("High",   2),
    "Disgust":  ("High",   2),
    "Fear":     ("High",   2),
    "Sad":      ("Medium", 1),
    "Surprise": ("Medium", 1),
    "Neutral":  ("Low",    0),
    "Happy":    ("Low",    0),
}

# ── Dual-branch fusion weights (from architecture diagram) ────────────────────
CNN_WEIGHT   = 0.20   # CNN branch contribution
PHYSIO_WEIGHT = 0.80  # Physio branch contribution

# ── Fused score → stress label thresholds ────────────────────────────────────
# Architecture diagram: Low (0–33) | Medium (34–66) | High (67–100)
STRESS_THRESHOLDS = {"low_max": 33, "medium_max": 66}

def fused_score_to_label(fused_score_0_1: float) -> tuple[str, int]:
    """
    Convert 0.0–1.0 fused score to stress label and numeric level.
    Fused score is scaled to 0–100 per diagram thresholds.

    Returns: (stress_label, stress_numeric)
      stress_label   : 'High' | 'Medium' | 'Low'
      stress_numeric : 2      | 1        | 0
    """
    score_100 = fused_score_0_1 * 100
    if score_100 <= STRESS_THRESHOLDS["low_max"]:
        return "Low", 0
    elif score_100 <= STRESS_THRESHOLDS["medium_max"]:
        return "Medium", 1
    else:
        return "High", 2


# ── Physio signal weights (architecture diagram, Physio branch) ───────────────
# 6 physiological signals extracted from MediaPipe 478 landmarks + iris
PHYSIO_WEIGHTS = {
    "blink_rate":    0.25,   # Most important stress indicator
    "eye_openness":  0.16,
    "gaze_jitter":   0.16,
    "head_motion":   0.12,
    "brow_tension":  0.08,
    "jaw_tension":   0.08,
}
# Note: weights sum to 0.85; remaining 0.15 distributed for normalization

# ── Core image loader ──────────────────────────────────────────────────────────
def load_fer2013_images(split: str = "train"):
    """
    Load all images from data/raw/<split>/<emotion>/ folders.

    Returns
    -------
    images : np.ndarray  shape (N, 48, 48)  dtype float32  range [0, 1]
    labels : np.ndarray  shape (N,)         dtype int32
    """
    split_dir = DATA_RAW / split
    if not split_dir.exists():
        raise FileNotFoundError(
            f"Folder not found: {split_dir}\n"
            f"Expected: data/raw/{split}/<emotion>/*.jpg"
        )

    images_list, labels_list = [], []

    for emotion_folder, label_idx in sorted(LABEL_MAP.items()):
        folder = split_dir / emotion_folder
        if not folder.exists():
            print(f"Skipping missing folder: {folder}")
            continue

        img_paths = sorted(
            p for p in folder.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )

        for img_path in img_paths:
            try:
                img = Image.open(img_path).convert("L")          # grayscale
                img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
                images_list.append(np.array(img, dtype=np.float32) / 255.0)
                labels_list.append(label_idx)
            except Exception as e:
                print(f"Skipped {img_path.name}: {e}")

        print(f"   {emotion_folder:<12} → {len(img_paths):>5} images  "
              f"(label {label_idx}, valence={EMOTION_VALENCE[EMOTION_LABELS[label_idx]]})")

    if not images_list:
        raise RuntimeError(f"No images loaded from {split_dir}.")

    images = np.stack(images_list)
    labels = np.array(labels_list, dtype=np.int32)
    print(f"\nLoaded [{split}]: {images.shape}  labels: {labels.shape}")
    return images, labels


def load_fer2013_train_val(val_split: float = 0.1, seed: int = 42):
    """Load training images and split off a validation set (10% default)."""
    print(f"Loading TRAIN images from: {DATA_RAW / 'train'}")
    images, labels = load_fer2013_images("train")

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(images))
    images, labels = images[idx], labels[idx]

    split_at = int(len(images) * (1 - val_split))
    train_images, train_labels = images[:split_at], labels[:split_at]
    val_images,   val_labels   = images[split_at:], labels[split_at:]

    print(f"\nTrain : {train_images.shape}")
    print(f"Val   : {val_images.shape}  ({val_split*100:.0f}% of train)")
    return train_images, train_labels, val_images, val_labels


def load_fer2013_test():
    """Load the test split (data/raw/test/)."""
    print(f"Loading TEST images from: {DATA_RAW / 'test'}")
    return load_fer2013_images("test")


# ── Save / load processed arrays ──────────────────────────────────────────────
def save_processed(images, labels, split_name: str):
    img_path = DATA_PROC / f"{split_name}_images.npy"
    lbl_path = DATA_PROC / f"{split_name}_labels.npy"
    np.save(img_path, images)
    np.save(lbl_path, labels)
    print(f"Saved {split_name}: {img_path.name}, {lbl_path.name}")


def load_processed(split_name: str):
    img_path = DATA_PROC / f"{split_name}_images.npy"
    lbl_path = DATA_PROC / f"{split_name}_labels.npy"
    if not img_path.exists():
        raise FileNotFoundError(
            f"Not found: {img_path}"
        )
    images = np.load(img_path)
    labels = np.load(lbl_path)
    print(f"Loaded {split_name}: {images.shape}, {labels.shape}")
    return images, labels


# ── Utility ───────────────────────────────────────────────────────────────────
def get_class_distribution(labels):
    unique, counts = np.unique(labels, return_counts=True)
    print("\nClass Distribution:")
    print(f"   {'Emotion':<12} {'Count':>6}  {'%':>6}  {'Valence':>8}")
    print("   " + "-" * 38)
    for cls, cnt in zip(unique, counts):
        pct  = cnt / len(labels) * 100
        name = EMOTION_LABELS.get(int(cls), str(cls))
        val  = EMOTION_VALENCE.get(name, 0.0)
        print(f"   {name:<12} {cnt:>6}  {pct:>5.1f}%  {val:>8.2f}")
    print()


def emotion_to_stress(emotion_idx: int):
    """Convert emotion class index → (stress_label, stress_numeric)."""
    name = EMOTION_LABELS.get(emotion_idx, "Neutral")
    return STRESS_MAP.get(name, ("Low", 0))


def compute_emotion_stress_score(emotion_idx: int, confidence: float) -> float:
    """
    CNN Branch output: weighted emotion stress score (0.0–1.0).
    Formula: valence × confidence
    This represents the CNN branch's contribution before fusion.
    """
    name    = EMOTION_LABELS.get(emotion_idx, "Neutral")
    valence = EMOTION_VALENCE.get(name, 0.0)
    return valence * confidence


if __name__ == "__main__":
    train_imgs, train_lbls, val_imgs, val_lbls = load_fer2013_train_val()
    test_imgs,  test_lbls  = load_fer2013_test()

    get_class_distribution(train_lbls)

    save_processed(train_imgs, train_lbls, "train_raw")
    save_processed(val_imgs,   val_lbls,   "val_raw")
    save_processed(test_imgs,  test_lbls,  "test_raw")
    print("\nAll splits saved to data/processed/")

import os
import sys
import json
import numpy as np
import tensorflow as tf
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data_loader import DATA_PROC, NUM_CLASSES
from preprocess  import load_preprocessed_as_dataset, BATCH_SIZE
from model       import build_emotion_cnn, compile_model, get_callbacks, MODEL_PATH

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)


def setup_gpu():
    """Configure TF GPU with memory growth; graceful CPU fallback."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU(s) detected: {len(gpus)}")
            for g in gpus:
                print(f"   → {g.name}")
        except RuntimeError as e:
            print(f"GPU config error: {e}")
    else:
        print(" No GPU — training on CPU (will be slower)")

    print(f"   TensorFlow  : {tf.__version__}")
    print(f"   Built w/ CUDA: {tf.test.is_built_with_cuda()}")
    print(f"\nArchitecture: CNN Branch (20% weight in dual-branch fusion)")
    print(f"   Physio branch (80%) uses MediaPipe — no training needed")


def load_data():
    """Load preprocessed train/val .npy files as tf.data.Dataset."""
    print("\nLoading preprocessed data...")
    train_ds = load_preprocessed_as_dataset("train", augment=True,  batch_size=BATCH_SIZE)
    val_ds   = load_preprocessed_as_dataset("val",   augment=False, batch_size=BATCH_SIZE)

    n_train = len(np.load(DATA_PROC / "train_images_proc.npy"))
    n_val   = len(np.load(DATA_PROC / "val_images_proc.npy"))
    print(f"   Train samples : {n_train}")
    print(f"   Val samples   : {n_val}")
    return train_ds, val_ds, n_train, n_val


def prepare_model():
    """Build, compile, and display model summary."""
    print("\nBuilding CNN branch model...")
    model = build_emotion_cnn()
    model = compile_model(model, learning_rate=1e-3)
    model.summary(line_length=80)
    return model


def train(model, train_ds, val_ds, n_train, n_val, epochs=50):
    """Train the CNN branch model."""
    steps_per_epoch  = n_train // BATCH_SIZE
    validation_steps = n_val   // BATCH_SIZE

    print(f"\nStarting CNN branch training...")
    print(f"   Epochs (max)   : {epochs}")
    print(f"   Batch size     : {BATCH_SIZE}")
    print(f"   Steps/epoch    : {steps_per_epoch}")
    print(f"   Best model path: {MODEL_PATH}\n")

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=get_callbacks(MODEL_PATH),
        verbose=1,
    )
    return history


def save_history(history):
    """Save training history dict to JSON."""
    hist_path = OUTPUTS_DIR / "training_history.json"
    hist_serializable = {
        k: [float(v) for v in vals]
        for k, vals in history.history.items()
    }
    with open(hist_path, "w") as f:
        json.dump(hist_serializable, f, indent=2)
    print(f"\nTraining history saved → {hist_path}")


def main():
    print("=" * 60)
    print("  Dual-Branch Stress Detection — CNN Branch Training")
    print("=" * 60)

    setup_gpu()
    train_ds, val_ds, n_train, n_val = load_data()
    model   = prepare_model()
    history = train(model, train_ds, val_ds, n_train, n_val, epochs=50)
    save_history(history)

    print("\nCNN branch training complete!")
    print(f"   Best model → {MODEL_PATH}")
    print(f"   History   → outputs/training_history.json")
    print("\nNext: Run 04_evaluation.ipynb")
    print("Then: Run 05_realtime_detection.ipynb for dual-branch live demo")


if __name__ == "__main__":
    main()

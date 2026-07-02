import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from data_loader import IMG_SIZE, NUM_CLASSES

MODEL_DIR  = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "stress_emotion_model.h5"

# ── CNN Branch Architecture ────────────────────────────────────────────────────
def build_emotion_cnn(
    input_shape=(IMG_SIZE, IMG_SIZE, 1),
    num_classes=NUM_CLASSES,
    dropout_rate=0.4,
    l2_lambda=1e-4,
) -> keras.Model:
    """
    Build the CNN branch model.

    Block structure (matches diagram: 32→64→128→256 filters):
      Block 1: Conv 32  → BN → ReLU → Conv 32  → BN → ReLU → MaxPool → Dropout
      Block 2: Conv 64  → BN → ReLU → Conv 64  → BN → ReLU → MaxPool → Dropout
      Block 3: Conv 128 → BN → ReLU → Conv 128 → BN → ReLU → MaxPool → Dropout
      Block 4: Conv 256 → BN → ReLU → GlobalAveragePool
      Head   : Dense 256 → BN → ReLU → Dropout → Dense 7 → Softmax

    Parameters
    ----------
    input_shape  : (H, W, C) — (48, 48, 1) for FER2013
    num_classes  : 7 (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral)
    dropout_rate : fraction of neurons dropped during training
    l2_lambda    : L2 weight decay coefficient

    Returns
    -------
    keras.Model (uncompiled)
    """
    inp = keras.Input(shape=input_shape, name="face_input")

    # ── Block 1: 32 filters ──────────────────────────────────────────────────
    x = layers.Conv2D(32, (3, 3), padding="same", name="conv1_1")(inp)
    x = layers.BatchNormalization(name="bn1_1")(x)
    x = layers.Activation("relu", name="relu1_1")(x)

    x = layers.Conv2D(32, (3, 3), padding="same", name="conv1_2")(x)
    x = layers.BatchNormalization(name="bn1_2")(x)
    x = layers.Activation("relu", name="relu1_2")(x)

    x = layers.MaxPooling2D((2, 2), strides=2, name="pool1")(x)   # 48→24
    x = layers.Dropout(dropout_rate * 0.5, name="drop1")(x)

    # ── Block 2: 64 filters ──────────────────────────────────────────────────
    x = layers.Conv2D(64, (3, 3), padding="same", name="conv2_1")(x)
    x = layers.BatchNormalization(name="bn2_1")(x)
    x = layers.Activation("relu", name="relu2_1")(x)

    x = layers.Conv2D(64, (3, 3), padding="same", name="conv2_2")(x)
    x = layers.BatchNormalization(name="bn2_2")(x)
    x = layers.Activation("relu", name="relu2_2")(x)

    x = layers.MaxPooling2D((2, 2), strides=2, name="pool2")(x)   # 24→12
    x = layers.Dropout(dropout_rate * 0.75, name="drop2")(x)

    # ── Block 3: 128 filters ─────────────────────────────────────────────────
    x = layers.Conv2D(128, (3, 3), padding="same", name="conv3_1")(x)
    x = layers.BatchNormalization(name="bn3_1")(x)
    x = layers.Activation("relu", name="relu3_1")(x)

    x = layers.Conv2D(128, (3, 3), padding="same", name="conv3_2")(x)
    x = layers.BatchNormalization(name="bn3_2")(x)
    x = layers.Activation("relu", name="relu3_2")(x)

    x = layers.MaxPooling2D((2, 2), strides=2, name="pool3")(x)   # 12→6
    x = layers.Dropout(dropout_rate, name="drop3")(x)

    # ── Block 4: 256 filters ─────────────────────────────────────────────────
    x = layers.Conv2D(256, (3, 3), padding="same", name="conv4_1")(x)
    x = layers.BatchNormalization(name="bn4_1")(x)
    x = layers.Activation("relu", name="relu4_1")(x)

    x = layers.GlobalAveragePooling2D(name="gap")(x)               # 6×6×256 → 256

    # ── Classification Head ───────────────────────────────────────────────────
    x = layers.Dense(256,
                     kernel_regularizer=regularizers.l2(l2_lambda),
                     name="dense1")(x)
    x = layers.BatchNormalization(name="bn_dense")(x)
    x = layers.Activation("relu", name="relu_dense")(x)
    x = layers.Dropout(dropout_rate, name="drop_dense")(x)

    out = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs=inp, outputs=out, name="EmotionCNN_Branch")
    return model


def compile_model(model: keras.Model, learning_rate: float = 1e-3) -> keras.Model:
    """Compile with Adam + categorical_crossentropy (one-hot labels)."""
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.TopKCategoricalAccuracy(k=3, name="top3_acc"),
        ],
    )
    return model


def get_callbacks(model_path=MODEL_PATH):
    """
    Training callbacks:
      ModelCheckpoint  : saves best model (monitors val_accuracy)
      ReduceLROnPlateau: halves LR when val_loss stalls for 5 epochs
      EarlyStopping    : stops if no improvement for 10 epochs
    """
    return [
        keras.callbacks.ModelCheckpoint(
            filepath=str(model_path),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
    ]


def load_model(model_path=MODEL_PATH) -> keras.Model:
    """Load saved model from disk."""
    print(f"Loading CNN model from {model_path}")
    return keras.models.load_model(str(model_path))


# ── Quick test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = build_emotion_cnn()
    model = compile_model(model)
    model.summary()

    dummy  = tf.random.uniform((4, IMG_SIZE, IMG_SIZE, 1))
    output = model(dummy, training=False)
    print(f"\n CNN Branch forward pass OK → output shape: {output.shape}")
    print(f"   Sample softmax: {output[0].numpy().round(3)}")
    print(f"\n   CNN branch weight in fusion: 20%")
    print(f"   Physio branch weight:        80%  (see physio_branch.py)")

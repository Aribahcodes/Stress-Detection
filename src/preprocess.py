import numpy as np
import tensorflow as tf
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from data_loader import NUM_CLASSES, IMG_SIZE, DATA_PROC

AUTOTUNE    = tf.data.AUTOTUNE
BATCH_SIZE  = 64
BUFFER_SIZE = 10000


def normalize_images(images: np.ndarray) -> np.ndarray:
    """Ensure images are float32 in [0, 1]."""
    images = images.astype(np.float32)
    if images.max() > 1.0:
        images /= 255.0
    return images


def add_channel_dim(images: np.ndarray) -> np.ndarray:
    """Add channel dimension: (N,48,48) → (N,48,48,1) for CNN input."""
    if images.ndim == 3:
        images = images[..., np.newaxis]
    return images


def one_hot_encode(labels: np.ndarray, num_classes: int = NUM_CLASSES) -> np.ndarray:
    """Convert integer labels → one-hot float32 arrays (vectorized)."""
    return np.eye(num_classes, dtype=np.float32)[labels]


def build_tf_dataset(
    images: np.ndarray,
    labels: np.ndarray,
    augment: bool = False,
    batch_size: int = BATCH_SIZE,
) -> tf.data.Dataset:
    """
    Build a fast tf.data.Dataset pipeline.

    Pipeline:
      images (float32, [0,1], shape HxWx1)
        → optional augmentation (flip + brightness)
        → batch
        → prefetch (overlaps GPU compute with CPU data prep)
    """
    ds = tf.data.Dataset.from_tensor_slices((images, labels))

    if augment:
        ds = ds.shuffle(BUFFER_SIZE)

        @tf.function
        def augment_fn(img, lbl):
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_brightness(img, max_delta=0.15)
            img = tf.clip_by_value(img, 0.0, 1.0)
            return img, lbl

        ds = ds.map(augment_fn, num_parallel_calls=AUTOTUNE)

    ds = ds.batch(batch_size)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def full_preprocess_pipeline(
    images_raw: np.ndarray,
    labels_raw: np.ndarray,
    split_name: str,
    augment: bool = False,
    save: bool = True,
):
    """
    End-to-end preprocessing for one dataset split.

    Steps:
      1. Normalize to [0,1]
      2. Add channel dim (→ HxWx1)
      3. One-hot encode labels
      4. Save .npy (optional)
      5. Return tf.data.Dataset

    Returns
    -------
    dataset     : tf.data.Dataset
    images_proc : np.ndarray (N, 48, 48, 1)  float32
    labels_ohe  : np.ndarray (N, NUM_CLASSES) float32
    """
    print(f"\nPreprocessing [{split_name}]  ({len(images_raw)} samples)")

    images = normalize_images(images_raw)
    images = add_channel_dim(images)
    labels = one_hot_encode(labels_raw)

    print(f"   images: {images.shape} dtype={images.dtype} "
          f"min={images.min():.3f} max={images.max():.3f}")
    print(f"   labels: {labels.shape} (one-hot, {NUM_CLASSES} classes)")

    if save:
        np.save(DATA_PROC / f"{split_name}_images_proc.npy", images)
        np.save(DATA_PROC / f"{split_name}_labels_ohe.npy",  labels)
        print(f"Saved to data/processed/{split_name}_*.npy")

    dataset = build_tf_dataset(images, labels, augment=augment)
    print(f"Dataset ready  (augment={augment}, batch_size={BATCH_SIZE})")

    return dataset, images, labels


def load_preprocessed_as_dataset(
    split_name: str,
    augment: bool = False,
    batch_size: int = BATCH_SIZE,
):
    """Load previously saved .npy files → tf.data.Dataset."""
    images = np.load(DATA_PROC / f"{split_name}_images_proc.npy")
    labels = np.load(DATA_PROC / f"{split_name}_labels_ohe.npy")
    print(f"Loaded preprocessed [{split_name}]: images={images.shape}")
    return build_tf_dataset(images, labels, augment=augment, batch_size=batch_size)


if __name__ == "__main__":
    dummy_images = np.random.randint(0, 256, (100, 48, 48), dtype=np.uint8)
    dummy_labels = np.random.randint(0, NUM_CLASSES, (100,), dtype=np.int32)

    ds, imgs, lbls = full_preprocess_pipeline(
        dummy_images, dummy_labels, "test_run", augment=False, save=False
    )
    for x, y in ds.take(1):
        print(f"\nBatch shape — images: {x.shape}, labels: {y.shape}")

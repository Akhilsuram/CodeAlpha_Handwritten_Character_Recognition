
from pathlib import Path
import numpy as np
import tensorflow as tf

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = DATA_DIR / "mnist_processed.npz"

print("=" * 60)
print("MNIST DATA PREPROCESSING")
print("=" * 60)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize pixel values to the range 0-1
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Add the channel dimension for CNN input
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

# Save the processed dataset
np.savez_compressed(
    OUTPUT_PATH,
    x_train=x_train,
    y_train=y_train,
    x_test=x_test,
    y_test=y_test,
)

print(f"Training images shape: {x_train.shape}")
print(f"Training labels shape: {y_train.shape}")
print(f"Testing images shape: {x_test.shape}")
print(f"Testing labels shape: {y_test.shape}")

print(f"Normalized pixel range: {x_train.min():.1f} to {x_train.max():.1f}")
print(f"Processed dataset saved to: {OUTPUT_PATH}")
print("Preprocessing completed successfully.")

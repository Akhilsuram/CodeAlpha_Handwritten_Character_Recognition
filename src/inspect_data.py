
from pathlib import Path
import tensorflow as tf
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("MNIST DATASET INSPECTION")
print("=" * 60)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

print(f"Training images shape: {x_train.shape}")
print(f"Training labels shape: {y_train.shape}")
print(f"Testing images shape:  {x_test.shape}")
print(f"Testing labels shape:  {y_test.shape}")

print(f"\nImage dimensions: {x_train.shape[1]} x {x_train.shape[2]}")
print(f"Number of classes: {len(np.unique(y_train))}")
print(f"Classes: {np.unique(y_train)}")
print(f"Pixel range: {x_train.min()} to {x_train.max()}")
print(f"Training samples: {len(x_train)}")
print(f"Testing samples: {len(x_test)}")

print("\nTraining class distribution:")
for digit, count in zip(*np.unique(y_train, return_counts=True)):
    print(f"Digit {digit}: {count}")

print("\nDataset inspection completed successfully.")

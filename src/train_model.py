
from pathlib import Path
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "mnist_processed.npz"
MODEL_DIR = BASE_DIR / "models"
PLOT_DIR = BASE_DIR / "outputs" / "plots"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "handwritten_digit_cnn.keras"

print("=" * 60)
print("CNN MODEL TRAINING")
print("=" * 60)

data = np.load(DATA_PATH)

x_train = data["x_train"]
y_train = data["y_train"]

x_test = data["x_test"]
y_test = data["y_test"]

# CNN architecture
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),

    tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(10, activation="softmax"),
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=2,
        restore_best_weights=True,
    ),
    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(MODEL_PATH),
        monitor="val_accuracy",
        save_best_only=True,
    ),
]

history = model.fit(
    x_train,
    y_train,
    epochs=10,
    batch_size=128,
    validation_split=0.1,
    callbacks=callbacks,
    verbose=1,
)

# Evaluate on the separate test set
test_loss, test_accuracy = model.evaluate(
    x_test, y_test, verbose=0
)

print(f"\nTest loss: {test_loss:.4f}")
print(f"Test accuracy: {test_accuracy:.4%}")
print(f"Model saved to: {MODEL_PATH}")

# Save training history
plt.figure(figsize=(10, 5))
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("CNN Training and Validation Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(PLOT_DIR / "01_training_accuracy.png", dpi=150)
plt.close()

plt.figure(figsize=(10, 5))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("CNN Training and Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(PLOT_DIR / "02_training_loss.png", dpi=150)
plt.close()

print("Training plots saved successfully.")
print("CNN training completed successfully.")

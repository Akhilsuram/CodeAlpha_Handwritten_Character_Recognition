
from pathlib import Path
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "mnist_processed.npz"
MODEL_PATH = BASE_DIR / "models" / "handwritten_digit_cnn.keras"
PLOT_DIR = BASE_DIR / "outputs" / "plots"

PLOT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("VISUALIZING SAMPLE PREDICTIONS")
print("=" * 60)

data = np.load(DATA_PATH)
x_test = data["x_test"]
y_test = data["y_test"]

model = tf.keras.models.load_model(MODEL_PATH)

# Select 15 test images
rng = np.random.default_rng(42)
indices = rng.choice(len(x_test), size=15, replace=False)

images = x_test[indices]
actual_labels = y_test[indices]
probabilities = model.predict(images, verbose=0)

predicted_labels = np.argmax(probabilities, axis=1)
confidences = np.max(probabilities, axis=1)

fig, axes = plt.subplots(3, 5, figsize=(12, 8))

for i, ax in enumerate(axes.flat):
    ax.imshow(images[i].squeeze(), cmap="gray")
    color = "green" if predicted_labels[i] == actual_labels[i] else "red"

    ax.set_title(
        f"Actual: {actual_labels[i]}\n"
        f"Pred: {predicted_labels[i]} ({confidences[i]:.1%})",
        color=color,
        fontsize=9,
    )
    ax.axis("off")

plt.suptitle("CNN Handwritten Digit Predictions")
plt.tight_layout()

output_path = PLOT_DIR / "04_sample_predictions.png"
plt.savefig(output_path, dpi=150)
plt.close()

correct = int(np.sum(predicted_labels == actual_labels))

print(f"Images displayed: {len(indices)}")
print(f"Correct predictions: {correct}/{len(indices)}")
print(f"Visualization saved to: {output_path}")
print("Prediction visualization completed successfully.")

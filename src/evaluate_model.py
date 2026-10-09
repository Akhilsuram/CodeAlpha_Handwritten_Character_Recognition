
from pathlib import Path
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "mnist_processed.npz"
MODEL_PATH = BASE_DIR / "models" / "handwritten_digit_cnn.keras"
PLOT_DIR = BASE_DIR / "outputs" / "plots"
OUTPUT_DIR = BASE_DIR / "outputs"

PLOT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("CNN MODEL EVALUATION")
print("=" * 60)

data = np.load(DATA_PATH)
x_test = data["x_test"]
y_test = data["y_test"]

model = tf.keras.models.load_model(MODEL_PATH)

test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
probabilities = model.predict(x_test, batch_size=256, verbose=0)
y_pred = np.argmax(probabilities, axis=1)

print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4%}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        labels=list(range(10)),
        digits=4,
        zero_division=0,
    )
)

cm = confusion_matrix(y_test, y_pred, labels=list(range(10)))

plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=range(10),
    yticklabels=range(10),
)
plt.xlabel("Predicted Digit")
plt.ylabel("Actual Digit")
plt.title("MNIST CNN Confusion Matrix")
plt.tight_layout()
plt.savefig(PLOT_DIR / "03_confusion_matrix.png", dpi=150)
plt.close()

# Save evaluation metrics
with open(OUTPUT_DIR / "evaluation_report.txt", "w", encoding="utf-8") as file:
    file.write(f"Test Loss: {test_loss:.4f}\n")
    file.write(f"Test Accuracy: {test_accuracy:.4%}\n\n")
    file.write(
        classification_report(
            y_test,
            y_pred,
            labels=list(range(10)),
            digits=4,
            zero_division=0,
        )
    )

print(f"\nConfusion matrix saved to: {PLOT_DIR / '03_confusion_matrix.png'}")
print(f"Evaluation report saved to: {OUTPUT_DIR / 'evaluation_report.txt'}")
print("Model evaluation completed successfully.")


from pathlib import Path
import sys
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "mnist_processed.npz"
MODEL_PATH = BASE_DIR / "models" / "handwritten_digit_cnn.keras"

def predict_array(model, image_array):
    image_array = np.asarray(image_array, dtype=np.float32)

    if image_array.max() > 1:
        image_array = image_array / 255.0

    if image_array.shape == (28, 28):
        image_array = image_array[..., np.newaxis]

    image_array = image_array[np.newaxis, ...]

    probabilities = model.predict(image_array, verbose=0)[0]
    predicted_digit = int(np.argmax(probabilities))

    return predicted_digit, probabilities

def load_custom_image(image_path):
    image = Image.open(image_path).convert("L")
    image = ImageOps.autocontrast(image)

    # Convert dark writing on a light background to MNIST-like colors.
    image_array = np.asarray(image)

    if image_array.mean() > 127:
        image = ImageOps.invert(image)

    image = image.resize((28, 28), Image.Resampling.LANCZOS)
    image_array = np.asarray(image, dtype=np.float32)

    # Suppress faint background noise.
    image_array[image_array < 35] = 0

    return image_array / 255.0

def main():
    if not MODEL_PATH.exists():
        print(f"Model not found: {MODEL_PATH}")
        print("Run python src/train_model.py first.")
        return

    model = tf.keras.models.load_model(MODEL_PATH)

    if len(sys.argv) > 1:
        image_path = Path(sys.argv[1])

        if not image_path.is_absolute():
            image_path = Path.cwd() / image_path

        if not image_path.exists():
            print(f"Image not found: {image_path}")
            return

        image_array = load_custom_image(image_path)
        source = str(image_path)
    else:
        if not DATA_PATH.exists():
            print(f"Processed dataset not found: {DATA_PATH}")
            print("Run python src/preprocess.py first.")
            return

        data = np.load(DATA_PATH)
        index = 0
        image_array = data["x_test"][index]
        actual_digit = int(data["y_test"][index])
        source = f"MNIST test sample {index}"

    predicted_digit, probabilities = predict_array(model, image_array)

    print("=" * 45)
    print("HANDWRITTEN DIGIT PREDICTION")
    print("=" * 45)
    print(f"Image source: {source}")

    if len(sys.argv) == 1:
        print(f"Actual digit: {actual_digit}")

    print(f"Predicted digit: {predicted_digit}")
    print(f"Confidence: {probabilities[predicted_digit]:.2%}")

    print("\nProbabilities for each digit:")
    for digit, probability in enumerate(probabilities):
        print(f"Digit {digit}: {probability:.2%}")

if __name__ == "__main__":
    main()

# Handwritten Character Recognition using CNN

A deep learning project developed as part of the **CodeAlpha Machine Learning Internship**.

## Project Overview

This project uses a Convolutional Neural Network (CNN) to recognize handwritten digits from 0 to 9 using the MNIST dataset.

## Dataset

- Dataset: MNIST
- Training images: 60,000
- Testing images: 10,000
- Image dimensions: 28 × 28 pixels
- Number of classes: 10
- Pixel values normalized to the range 0–1

The dataset is loaded using `tf.keras.datasets.mnist`.

## Model Architecture

The CNN consists of:

- Convolutional layer with 32 filters
- Max pooling layer
- Convolutional layer with 64 filters
- Max pooling layer
- Flatten layer
- Dense layer with 128 neurons
- Dropout layer
- Output layer with 10 classes and softmax activation

## Results

The initial trained model achieved **99.08% test accuracy**.

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- Sample prediction visualizations

## Technologies

- Python 3.11
- TensorFlow / Keras
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Pillow

## Installation

Create and activate a virtual environment, then install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run the Project

Run the complete pipeline:

```powershell
python main.py
```

## Individual Scripts

```powershell
python src/inspect_data.py
python src/preprocess.py
python src/train_model.py
python src/evaluate_model.py
python src/visualize_predictions.py
python src/predict.py
```

To predict a digit from your own image, save a single handwritten digit as `my_digit.png` in the project root and run:

```powershell
python src/predict.py my_digit.png
```

## Project Structure

```text
CodeAlpha_Handwritten_Character_Recognition/
├── data/
├── models/
├── notebooks/
├── outputs/
│   ├── plots/
│   └── evaluation_report.txt
├── src/
│   ├── inspect_data.py
│   ├── preprocess.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   ├── visualize_predictions.py
│   └── predict.py
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Limitations

The model is trained on MNIST's small, grayscale, centered digit images. Predictions on real-world photographs or unusually styled handwriting may be less accurate.

This project is intended for educational purposes.

## Internship

**CodeAlpha Machine Learning Internship**

Task: Handwritten Character Recognition
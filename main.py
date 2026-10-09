
from pathlib import Path
import subprocess
import sys

BASE_DIR = Path(__file__).resolve().parent

SCRIPTS = [
    "src/inspect_data.py",
    "src/preprocess.py",
    "src/train_model.py",
    "src/evaluate_model.py",
    "src/visualize_predictions.py",
    "src/predict.py",
]

def main():
    print("=" * 60)
    print("HANDWRITTEN CHARACTER RECOGNITION PROJECT")
    print("=" * 60)

    for script in SCRIPTS:
        print(f"\nRunning: {script}")

        result = subprocess.run(
            [sys.executable, str(BASE_DIR / script)],
            cwd=BASE_DIR,
        )

        if result.returncode != 0:
            print(f"\nERROR: {script} failed.")
            sys.exit(result.returncode)

    print("\n" + "=" * 60)
    print("PROJECT COMPLETED SUCCESSFULLY")
    print("=" * 60)

if __name__ == "__main__":
    main()

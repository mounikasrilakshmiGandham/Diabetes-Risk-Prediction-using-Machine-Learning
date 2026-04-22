"""
download_dataset.py
===================
Downloads the Pima Indians Diabetes Dataset from a public URL
and saves it to the data/ directory.

Run this ONCE before training:
    python download_dataset.py
"""

import urllib.request
import os
import sys


DATASET_URL = (
    "https://raw.githubusercontent.com/jbrownlee/Datasets/master/"
    "pima-indians-diabetes.data.csv"
)

# Column names for this CSV (it has no header row)
COLUMN_NAMES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "Outcome",
]

OUTPUT_PATH = os.path.join("data", "diabetes.csv")


def download():
    os.makedirs("data", exist_ok=True)

    if os.path.exists(OUTPUT_PATH):
        print(f"[INFO] Dataset already exists at: {OUTPUT_PATH}")
        print("[INFO] Delete it and re-run this script to re-download.")
        return

    print(f"[INFO] Downloading dataset from:\n  {DATASET_URL}")
    print("[INFO] This requires an internet connection (one-time only).")

    try:
        urllib.request.urlretrieve(DATASET_URL, OUTPUT_PATH + ".tmp")
    except Exception as e:
        print(f"\n[ERROR] Download failed: {e}")
        print("\n  Manual download steps:")
        print("  1. Open: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database")
        print("  2. Download diabetes.csv")
        print("  3. Place it in the data/ folder of this project")
        sys.exit(1)

    # Add header row since the raw file has no column names
    import csv
    rows = []
    with open(OUTPUT_PATH + ".tmp", "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) == len(COLUMN_NAMES):
                rows.append(row)

    with open(OUTPUT_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(COLUMN_NAMES)   # write header
        writer.writerows(rows)

    os.remove(OUTPUT_PATH + ".tmp")

    print(f"\n[SUCCESS] Dataset saved to: {OUTPUT_PATH}")
    print(f"[INFO]    Rows downloaded: {len(rows)}")


if __name__ == "__main__":
    download()

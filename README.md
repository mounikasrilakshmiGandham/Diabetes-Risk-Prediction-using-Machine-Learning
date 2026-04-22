# Diabetes Prediction & Type Classification System

A machine learning system for diabetes screening and clinical type classification.

## Quick Start

```bash
# 1. Create virtual environment
python -m venv venv
venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download dataset
python download_dataset.py

# 4. Train model (run ONCE)
python train.py

# 5. Start web app
python app.py
# Then open: http://127.0.0.1:5000
```

## Architecture
- **Stage 1 (ML)**: Random Forest → Diabetic / Non-Diabetic
- **Stage 2 (Rules)**: ADA 2023 clinical rules → Type 1 / Type 2 / Gestational / Prediabetes

## Tech Stack
Python 3.10+ | Scikit-learn | Flask | Pandas | NumPy | Matplotlib | Seaborn

# Informal Market Product Recognition

This repository is an MVP scaffold for the Informal Market Product Recognition system.

Quick start (Windows PowerShell):

1. Create and activate virtual environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

Alternative (recommended): use Conda/Miniconda

```powershell
# Create and activate Conda environment with Python 3.11
conda create -n marketrec python=3.11 -y
conda activate marketrec
python --version  # should show 3.11.x
pip install --upgrade pip
pip install -r .\requirements.txt
```

Notes:
- TensorFlow currently provides prebuilt wheels for Python 3.8–3.11 on Windows; using a `conda` env with Python 3.11 avoids compatibility issues on newer system Python versions (e.g., 3.13).
- If `conda` is not available, install a system Python 3.11 and recreate a venv using that interpreter.

Training the model
```powershell
conda activate marketrec
python model/train.py --data-dir ./dataset --output-dir ./model --epochs 10 --batch-size 32 --img-size 224 224
```
Outputs: `model/product_classifier/` (SavedModel), `model/product_classifier.h5`, `model/labels.txt`.

Dataset structure (required)
```
dataset/
	train/
		banana/
		garlic/
		red_onion/
		berbere/
		...
```
Use `python dataset/stats.py --data-dir ./dataset/train` to see class counts.

Seed example prices
```powershell
conda activate marketrec
python -m backend.seed_prices
```

Run backend
```powershell
conda activate marketrec
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Run frontend (Streamlit)
```powershell
conda activate marketrec
cd C:\Users\mommy\Documents\market-recognition
streamlit run frontend/streamlit_app.py
```

2. Run the backend (from `market-recognition` root)

```powershell
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

3. Run the frontend demo (Streamlit)

```powershell
cd ..
streamlit run frontend/streamlit_app.py
```

What is included:
- Minimal `FastAPI` backend with `/predict` endpoint and simple SQLite persistence.
- `model/train.py` training stub using TensorFlow/Keras (transfer learning).
- `frontend/streamlit_app.py` simple demo UI to upload a photo, enter qty/vendor and call the API.

Next steps:
- Populate `dataset/` with labeled images (see `dataset/README-dataset.md`).
- Train model using `model/train.py`.
- Replace the dummy predictor in `backend/main.py` with a real model loader.
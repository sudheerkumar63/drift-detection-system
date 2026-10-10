# Deep Learning Model Degradation Detection and Adaptation for Traffic Sign Recognition Under Environmental Drift

This project is a demonstration and research dashboard for monitoring traffic sign recognition under environmental drift, simulating environmental degradation, estimating risk, and verifying safety for model adaptation.

## Project structure

- `frontend/` - React + Vite + Tailwind dashboard
- `backend/` - FastAPI API and service layer
- `data/` - SQLite database and sample data
- `model_weights/` - trained checkpoints and reference features
- `requirements.txt` - Python dependencies

## Quick start

### Python backend

```bash
python -m venv .venv
. .venv\Scripts\activate
pip install -r requirements.txt
python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev -- --host 0.0.0.0 --port 5173
```

Then open the frontend URL and configure the API endpoint if needed.

## Model training and availability

The repository does not include trained weights. Until training completes, the backend stays in transparent demo mode and marks its deterministic sample predictions as non-model results. Performance Analytics uses saved prediction records and reports checkpoint evaluation metrics only when available; it does not fill missing data with sample values.

The CNN uses a spatial feature head, class-balanced sampling, and image augmentation. If `data/gtsrb/Train/` contains class folders `00000` through `00042`, training uses an 80/20 stratified holdout from that local set; otherwise it downloads the torchvision GTSRB train and official test splits.

```powershell
.\.venv\Scripts\python.exe -m backend.train_model --epochs 15
```

The script saves the best validation checkpoint as `model_weights/gtsrb_model.pth`, and prints accuracy, evaluated sample count, and per-class accuracy. Evaluation reports identify whether they use the local stratified holdout or the official test split. Checkpoint updates are picked up by the backend when the file changes. A small smoke run can limit dataset sizes:

```powershell
.\.venv\Scripts\python.exe -m backend.train_model --epochs 1 --max-train-samples 512 --max-test-samples 256 --output $env:TEMP\gtsrb_smoke.pth
```

The active checkpoint is replaced only after training and validation finish. The inference service checks the architecture, preprocessing, class order, and completed-training marker. `/api/model-status`, `/api/health`, predictions, and Performance Analytics report the active mode and recorded validation accuracy. GTSRB is a cropped-sign benchmark, so accuracy on real photos with large backgrounds, occlusion, unusual angles, or different sign styles may be lower; this model is not a safety-certified road-sign detector.

## Degradation and safety verification

The Model Degradation page estimates heuristic risk from the latest prediction and drift-detection result. It requires both results and uses the thresholds shown in its controls; it does not substitute fixed sample inputs.

Safety Verification requires measured candidate validation metrics (`accuracy`, `precision`, `recall`, and `f1`) in the range 0–1. Enter values computed from a labeled validation set, adjust acceptance thresholds as needed, then run verification. Missing metrics are not inferred from prediction confidence or filled with demo values. Passing this check only means the submitted metrics met the selected thresholds; the dashboard does not independently validate those measurements and does not certify automotive safety.

## Sample endpoints

- `GET /api/health`
- `POST /api/predict`
- `POST /api/simulate-environment`
- `POST /api/compare-predictions`
- `POST /api/drift-detection`
- `POST /api/degradation-estimate`
- `POST /api/adaptation/start`
- `GET /api/adaptation/status/{job_id}`
- `POST /api/adaptation/stop/{job_id}`
- `GET /api/model-status`
- `POST /api/safety/verify`
- `GET /api/history`
- `GET /api/analytics`

## Research note

This is a demonstration system for research and educational purposes. It is not a certified automotive safety system.

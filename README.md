# 🚦 Drift Detection System — ML Model Monitoring

> Production MLOps system to detect data drift & model degradation for Traffic Sign Classification (GTSRB).

🔗 **Repo:** https://github.com/sudheerkumar63/drift-detection-system

### 🎯 Why?
ML models fail in production when data changes (night, rain, blur). This system detects drift and triggers alerts + retraining.

### ✨ Features
- Drift Detection: PSI, KS-Test, Evidently AI
- Real-time Dashboard: React + FastAPI
- Explainability: GradCAM for drifted images
- Dockerized & Production Ready

### 🚀 Quick Start
```bash
git clone https://github.com/sudheerkumar63/drift-detection-system.git
cd drift-detection-system
docker-compose up --build
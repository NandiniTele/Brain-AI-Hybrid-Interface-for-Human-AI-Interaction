# Brain-AI-Hybrid-Interface-for-Human-AI-Interaction
# BRAIN-AI-HYBRID-INTERFACE-SIMULATOR

![License](https://img.shields.io/badge/License-MIT-green) ![Python](https://img.shields.io/badge/Python-3.11%2B-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange) ![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688) ![React](https://img.shields.io/badge/React-18%2B-61DAFB)

A high-performance, software-based **Brain–AI Hybrid Interface Simulator** designed for computational neuroscience research, academic demonstration, and algorithmic benchmarking. It implements an end-to-end signal processing and deep learning inference pipeline featuring a **Multi-scale CNN-BiLSTM-Transformer** neural architecture, alongside an interactive real-time research dashboard.

> **Note on Scope**: This project is a **purely software-based EEG/BCI simulator and decoder**. It does not require physical EEG headsets, scalp electrode hardware, or live physiological acquisition devices. It operates using validated benchmark EEG data (PhysioNet EEGBCI) and real-time synthetic signal generators.

---

## 🚀 Key Capabilities

- **Software-Based Signal Pipeline**: MNE-Python based digital filtering (Band-pass 0.5–40 Hz, Notch 50 Hz, Z-score normalization, channel standardisation to 19 channels in the international 10–20 system).
- **Deep Learning Architecture**: PyTorch implementation of **Multi-scale CNN → BiLSTM → Transformer Encoder → Prediction Fusion → Classifier** predicting 5 cognitive/emotional states (Calm, Focused, Stressed, Fatigued, Excited) and 7 continuous regression indices (Focus, Attention, Stress, Fatigue, Cognitive Load, Valence, Arousal).
- **Explainable AI (XAI)**: Feature Attribution (Explainability Proxy) providing band-power relative importance across Delta, Theta, Alpha, Beta, and Gamma frequency bands.
- **Audited Dataset Analytics**: Validated on the **PhysioNet EEGBCI** dataset (3 subjects, 5 selected runs per subject, 15 clusters, 735 windows of 2-second duration sampled at 160 Hz, mapped from 64 original channels to 19 standard 10–20 channels).
- **Dual Persistence Architecture**: SQLite serves as the default local persistence layer for session logs and telemetry, with MongoDB supported as an optional database backend.
- **Real-Time Research Dashboard**: Interactive WebSocket telemetry streaming at ~25 FPS with brain region topographic mapping, 10–20 scalp map, time-series oscilloscope, and dynamic model metrics.

---

## 🧠 System Architecture

```
[Simulated EEG / PhysioNet 19-Ch @ 160Hz]
                │
                ▼
   [Preprocessing & Filtering]
   (0.5–40Hz Bandpass + 50Hz Notch + Z-Score)
                │
                ▼
   [Feature Extraction (19 × 27 = 513)]
   (PSD Bands, Hjorth Parameters, Statistical Moments)
                │
                ▼
   [Multi-Scale Residual 1D-CNN Blocks]
                │
                ▼
   [Bidirectional LSTM Sequence Modeling]
                │
                ▼
   [Transformer Encoder + Multi-Head Self-Attention]
                │
                ▼
   [Global Average Pooling & Prediction Heads]
   ├─ 5-Class Emotion Logits
   └─ 7-Head Cognitive Regression Indices
```

---

## 🛠️ Requirements & Setup

### Prerequisites
- **Python**: `>= 3.11` (Python 3.11.9 recommended)
- **Node.js**: `>= 18.0.0`
- **npm**: `>= 9.0.0`

### 1. Pre-Flight Verification
```bash
# Verify Python environment and required dependencies
npm run check
```

### 2. Full-Stack Development
```bash
# Install all dependencies (frontend + backend)
npm run install:all

# Run backend and frontend concurrently
npm run dev
```
- **Dashboard**: `http://localhost:5173`
- **API Documentation**: `http://localhost:8000/docs`
- **WebSocket Endpoint**: `ws://localhost:8000/ws`

### 3. Individual Component Launch
```bash
# Backend only (FastAPI)
cd backend
py -3.11 startup_check.py
py -3.11 -m uvicorn main:app --reload --port 8000

# Frontend only (Vite + React)
cd frontend
npm install
npm run dev
```

---

## 🧪 Testing

```bash
# Run backend pytest suite (11 unit and ML pipeline tests)
cd backend
py -3.11 -m pytest tests/
```

---

## 📡 API Reference

| Method | Endpoint | Description |
|---|---|---|
| `WS` | `/ws` | Real-time WebSocket streaming 19-ch EEG signals & predictions |
| `GET` | `/health` | System health check and database status |
| `GET` | `/datasets` | Registered dataset configurations & active metadata |
| `POST` | `/simulate` | Configure simulation stream and artifact injection |
| `POST` | `/train` | Trigger model training in non-blocking thread executor |
| `GET` | `/train-status` | Current training loop status and progress |
| `GET` | `/model-stats` | Dynamically calculated parameter count & architecture info |
| `GET` | `/model-comparison` | Stored benchmark comparison metrics |
| `GET` | `/logs` | SQLite paginated telemetry logs |
| `POST` | `/auth/token` | JWT authentication endpoint |

---

## 📂 Repository Structure

- `/backend`: FastAPI service, Multi-scale CNN-BiLSTM-Transformer model (`model.py`), dataset loaders, preprocessing, and test suite.
- `/frontend`: React + TypeScript research dashboard, oscilloscope, 10–20 scalp map, and model evaluation modules.
- `/weights`: Stored `.pth` model weights, configuration, and audited experiment metrics.
- `PROJECT_PAPER_CONSISTENCY_REPORT.md`: Authoritative reference report for manuscript alignment and reviewer responses.


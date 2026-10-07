"""
metrics_service.py — Single source of truth for all dashboard metrics.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional

import torch

from utils import setup_logger

logger = setup_logger("MetricsService")

N_CHANNELS = 19
FEATURES_PER_CH = 27
INPUT_FEATURES = N_CHANNELS * FEATURES_PER_CH

# Rolling telemetry for latency / FPS (shared across WS sessions)
_latency_history: List[float] = []
_fps_timestamps: List[float] = []


def _resolve_file_path(filename: str) -> Optional[str]:
    """Finds a weights/metrics file in backend/weights or root weights directories."""
    candidates = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights", filename),
        os.path.join(os.getcwd(), "backend", "weights", filename),
        os.path.join(os.getcwd(), "weights", filename),
        os.path.join("weights", filename),
        filename,
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None


WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")
METRICS_PATH = os.path.join(WEIGHTS_DIR, "metrics.json")
BENCHMARKS_PATH = os.path.join(WEIGHTS_DIR, "benchmarks.json")


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def format_param_count(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def load_metrics() -> Dict[str, Any]:
    path = _resolve_file_path("metrics.json")
    if path:
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception as exc:
            logger.error(f"Failed to load metrics from {path}: {exc}")
    return {}


def load_benchmarks() -> List[Dict[str, Any]]:
    path = _resolve_file_path("benchmarks.json")
    if path:
        try:
            with open(path, "r") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
        except Exception as exc:
            logger.error(f"Failed to load benchmarks from {path}: {exc}")
    return []


def get_device_info() -> Dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    info: Dict[str, Any] = {
        "cuda_available": cuda_available,
        "device_label": "CUDA" if cuda_available else "CPU",
        "device_name": torch.cuda.get_device_name(0) if cuda_available else "Running on CPU",
        "framework": f"PyTorch {torch.__version__.split('+')[0]}",
    }
    return info


def get_gpu_usage() -> float:
    """Return GPU utilization % or 0.0 when CUDA / pynvml is unavailable."""
    if not torch.cuda.is_available():
        return 0.0
    try:
        if hasattr(torch.cuda, "utilization"):
            return float(torch.cuda.utilization())
    except Exception:
        pass
    return 0.0


def record_latency(ms: float) -> None:
    global _latency_history
    if ms <= 0 or ms != ms:  # NaN guard
        return
    _latency_history.append(ms)
    if len(_latency_history) > 200:
        _latency_history = _latency_history[-200:]


def get_latency_stats(current_ms: float) -> Dict[str, float]:
    history = _latency_history or ([current_ms] if current_ms > 0 else [0.0])
    return {
        "current_latency": round(current_ms, 2),
        "avg_latency": round(sum(history) / len(history), 2),
        "max_latency": round(max(history), 2),
    }


def record_fps_tick() -> float:
    """Record a WS loop tick and return measured FPS."""
    global _fps_timestamps
    now = time.time()
    _fps_timestamps.append(now)
    if len(_fps_timestamps) > 60:
        _fps_timestamps = _fps_timestamps[-60:]
    if len(_fps_timestamps) < 2:
        return 25.0
    elapsed = _fps_timestamps[-1] - _fps_timestamps[0]
    if elapsed <= 0:
        return 25.0
    return round((len(_fps_timestamps) - 1) / elapsed, 1)


def get_model_stats(engine) -> Dict[str, Any]:
    metrics = load_metrics()
    device_info = get_device_info()
    
    param_count = "N/A"
    raw_param_count = None
    if engine and hasattr(engine, "model") and engine.model is not None:
        raw_param_count = count_parameters(engine.model)
        param_count = format_param_count(raw_param_count)

    acc = metrics.get("accuracy")
    prec = metrics.get("precision")
    rec = metrics.get("recall")
    f1 = metrics.get("f1_score")
    epoch = metrics.get("epoch")
    cv = metrics.get("cross_val_score")
    roc = metrics.get("roc_auc")
    t_sec = metrics.get("training_time_sec")

    arch = getattr(engine, "architecture", "Multi-scale CNN-BiLSTM-Transformer")
    arch_full = getattr(engine, "architecture_full", "Multi-scale CNN → BiLSTM → Transformer Encoder → Prediction Fusion → Classifier")
    version = getattr(engine, "version", "v5.0.0-PROD")
    is_loaded = getattr(engine, "is_loaded", False)

    return {
        "architecture": arch,
        "architecture_full": arch_full,
        "version": version,
        "accuracy": round(float(acc), 2) if acc is not None else None,
        "precision": round(float(prec), 2) if prec is not None else None,
        "recall": round(float(rec), 2) if rec is not None else None,
        "f1_score": round(float(f1), 2) if f1 is not None else None,
        "epoch": str(epoch) if epoch is not None else "N/A",
        "cross_val_score": round(float(cv), 2) if cv is not None else None,
        "roc_auc": round(float(roc), 4) if roc is not None else None,
        "cohens_kappa": round(float(metrics.get("cohens_kappa")), 4) if metrics.get("cohens_kappa") is not None else None,
        "matthews_corrcoef": round(float(metrics.get("matthews_corrcoef")), 4) if metrics.get("matthews_corrcoef") is not None else None,
        "training_time_sec": round(float(t_sec), 1) if t_sec is not None else None,
        "parameters": param_count,
        "raw_parameters": raw_param_count,
        "framework": device_info["framework"],
        "device": device_info["device_label"],
        "device_name": device_info["device_name"],
        "cuda_available": device_info["cuda_available"],
        "status": "INFERENCING (ACTIVE)" if is_loaded else "STANDBY",
        "source": "Current Experiment" if metrics else "No Experiment Data",
    }


def get_model_comparison() -> List[Dict[str, Any]]:
    """
    Returns only REAL stored experiment results from metrics or benchmarks file.
    Does NOT invent or calculate fake baselines.
    """
    metrics = load_metrics()
    comparison = metrics.get("model_comparison")
    if comparison and isinstance(comparison, list) and len(comparison) > 0:
        return comparison

    benchmarks = load_benchmarks()
    if benchmarks and isinstance(benchmarks, list) and len(benchmarks) > 0:
        return benchmarks

    return []


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        v = float(value)
        if v != v:  # NaN
            return default
        return v
    except (TypeError, ValueError):
        return default


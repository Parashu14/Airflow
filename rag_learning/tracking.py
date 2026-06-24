from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_mlflow = None
try:
    import mlflow as _mlflow
except ImportError:
    pass

# Newer MLflow versions reject filesystem backends by default.
if _mlflow:
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")


def enabled() -> bool:
    return _mlflow is not None


def start_run(
    run_name: str | None = None,
    experiment_name: str | None = None,
    tags: dict[str, str] | None = None,
    nested: bool = False,
) -> None:
    if not _mlflow:
        return
    try:
        if experiment_name:
            _mlflow.set_experiment(experiment_name)
        _mlflow.start_run(run_name=run_name, nested=nested)
        if tags:
            _mlflow.set_tags(tags)
    except Exception as exc:
        logger.warning("MLflow start_run failed (tracking disabled): %s", exc)
        _disable()


def log_params(params: dict[str, Any]) -> None:
    if _mlflow:
        try:
            _mlflow.log_params(params)
        except Exception:
            pass


def log_metrics(metrics: dict[str, float], step: int | None = None) -> None:
    if _mlflow:
        try:
            _mlflow.log_metrics(metrics, step=step)
        except Exception:
            pass


def log_artifact(path: str | Path) -> None:
    if _mlflow:
        try:
            _mlflow.log_artifact(str(path))
        except Exception:
            pass


def log_text(text: str, artifact_path: str) -> None:
    if _mlflow:
        try:
            _mlflow.log_text(text, artifact_path)
        except Exception:
            pass


def end_run(status: str = "FINISHED") -> None:
    if _mlflow:
        try:
            _mlflow.end_run(status=status)
        except Exception:
            pass


def set_tracking_uri(uri: str) -> None:
    if _mlflow:
        try:
            _mlflow.set_tracking_uri(uri)
        except Exception as exc:
            logger.warning("MLflow set_tracking_uri failed: %s", exc)


def active_run() -> Any:
    if _mlflow:
        try:
            return _mlflow.active_run()
        except Exception:
            return None
    return None


def _disable() -> None:
    global _mlflow
    _mlflow = None

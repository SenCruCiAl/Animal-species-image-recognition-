"""Model loading utilities for the animal species classifier service."""

from __future__ import annotations

import json
import os
from pathlib import Path
from threading import Lock
from typing import List, Tuple

from tensorflow.keras.models import load_model

DEFAULT_MODEL_PATH = Path(os.getenv("MODEL_PATH", "model.h5"))
DEFAULT_CLASS_MAP_PATH = Path(os.getenv("CLASS_MAP_PATH", "class_indices.json"))

_model = None
_class_names: List[str] | None = None
_model_lock = Lock()


class ModelLoadError(RuntimeError):
    """Raised when model assets are unavailable or invalid."""


def _load_class_names(class_map_path: Path) -> List[str]:
    if not class_map_path.exists():
        raise ModelLoadError(
            f"Class map file not found at '{class_map_path}'. "
            "Train a model first to generate class_indices.json."
        )

    with class_map_path.open("r", encoding="utf-8") as f:
        class_indices = json.load(f)

    if not isinstance(class_indices, dict) or not class_indices:
        raise ModelLoadError("class_indices.json must contain a non-empty object.")

    return [name for name, _ in sorted(class_indices.items(), key=lambda item: item[1])]


def get_model_and_classes(
    model_path: Path = DEFAULT_MODEL_PATH,
    class_map_path: Path = DEFAULT_CLASS_MAP_PATH,
) -> Tuple[object, List[str]]:
    """Lazily load and cache the trained Keras model and class names."""
    global _model, _class_names

    if _model is not None and _class_names is not None:
        return _model, _class_names

    with _model_lock:
        if _model is None or _class_names is None:
            if not model_path.exists():
                raise ModelLoadError(
                    f"Trained model not found at '{model_path}'. "
                    "Run training first to generate model.h5."
                )
            _model = load_model(model_path)
            _class_names = _load_class_names(class_map_path)

    return _model, _class_names

"""Inference pipeline for uploaded images."""

from __future__ import annotations

from io import BytesIO
from typing import List, Tuple

import numpy as np
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

IMAGE_SIZE = (224, 224)


class InferenceError(ValueError):
    """Raised for invalid or unsupported input images."""


def preprocess_image(image_bytes: bytes, target_size: tuple[int, int] = IMAGE_SIZE) -> np.ndarray:
    """Convert raw image bytes into a preprocessed model input batch."""
    try:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:  # Pillow raises multiple exception types for bad input.
        raise InferenceError("Uploaded file is not a valid image.") from exc

    image = image.resize(target_size)
    image_array = np.array(image, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)
    return preprocess_input(image_array)


def predict_species(model: object, class_names: List[str], image_bytes: bytes) -> Tuple[str, float]:
    """Run model inference and return top class with confidence."""
    if not class_names:
        raise InferenceError("Class names list cannot be empty.")

    processed = preprocess_image(image_bytes)
    predictions = model.predict(processed, verbose=0)

    if predictions.ndim != 2 or predictions.shape[1] != len(class_names):
        raise InferenceError("Model output shape does not match class mapping.")

    top_idx = int(np.argmax(predictions[0]))
    confidence = float(predictions[0][top_idx])
    return class_names[top_idx], confidence

from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image

from app.inference import predict_species


class DummyModel:
    def predict(self, image_batch, verbose=0):
        assert image_batch.shape == (1, 224, 224, 3)
        return np.array([[0.1, 0.8, 0.1]], dtype=np.float32)


def _create_test_image_bytes() -> bytes:
    image = Image.new("RGB", (300, 300), color=(120, 180, 200))
    buf = BytesIO()
    image.save(buf, format="JPEG")
    return buf.getvalue()


def test_predict_species_returns_top_class_and_confidence():
    model = DummyModel()
    class_names = ["cat", "dog", "elephant"]
    img_bytes = _create_test_image_bytes()

    predicted_class, confidence = predict_species(model, class_names, img_bytes)

    assert predicted_class == "dog"
    assert confidence == 0.8

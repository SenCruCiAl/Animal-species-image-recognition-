"""Dataset loading utilities for transfer-learning training pipeline."""

from __future__ import annotations

from pathlib import Path

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def create_data_generators(
    dataset_dir: str,
    image_size: tuple[int, int] = (224, 224),
    batch_size: int = 32,
    validation_split: float = 0.2,
):
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists() or not dataset_path.is_dir():
        raise FileNotFoundError(f"Dataset directory '{dataset_dir}' was not found.")

    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        validation_split=validation_split,
        rotation_range=25,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest",
    )

    eval_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        validation_split=validation_split,
    )

    train_gen = train_datagen.flow_from_directory(
        dataset_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode="categorical",
        subset="training",
        shuffle=True,
    )

    val_gen = eval_datagen.flow_from_directory(
        dataset_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
    )

    return train_gen, val_gen

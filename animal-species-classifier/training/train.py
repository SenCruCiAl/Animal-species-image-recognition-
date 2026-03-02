"""Train an animal species classifier with MobileNetV2 transfer learning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam

from training.dataset_loader import create_data_generators


def build_model(num_classes: int, fine_tune_at: int = 120) -> Model:
    base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = True

    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation="softmax")(x)
    model = Model(inputs, outputs)

    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MobileNetV2 animal classifier")
    parser.add_argument("--dataset-dir", required=True, help="Path to dataset directory")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output-model", default="model.h5")
    parser.add_argument("--output-class-map", default="class_indices.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_gen, val_gen = create_data_generators(
        dataset_dir=args.dataset_dir,
        batch_size=args.batch_size,
    )

    model = build_model(num_classes=train_gen.num_classes)

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=2, min_lr=1e-6),
    ]

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    output_model = Path(args.output_model)
    output_model.parent.mkdir(parents=True, exist_ok=True)
    model.save(output_model)

    class_map_path = Path(args.output_class_map)
    class_map_path.parent.mkdir(parents=True, exist_ok=True)
    with class_map_path.open("w", encoding="utf-8") as f:
        json.dump(train_gen.class_indices, f, indent=2)

    print(f"Model saved to {output_model.resolve()}")
    print(f"Class map saved to {class_map_path.resolve()}")


if __name__ == "__main__":
    main()

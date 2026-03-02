# Animal Species Image Recognition App

Production-ready web app for animal species classification using **MobileNetV2 transfer learning**, **FastAPI**, and a simple HTML frontend.

## Features
- Transfer learning on top of MobileNetV2 (ImageNet pretrained).
- Custom softmax classifier head for user-defined animal classes.
- Data augmentation with `ImageDataGenerator`.
- FastAPI backend with file upload inference endpoint.
- Browser-based upload UI.
- Unit tests with pytest.
- GitHub Actions CI pipeline.

## Project Structure

```text
animal-species-classifier/
├── app/
│   ├── main.py
│   ├── model_loader.py
│   ├── inference.py
│   └── templates/index.html
├── training/
│   ├── train.py
│   └── dataset_loader.py
├── tests/
│   └── test_inference.py
├── requirements.txt
├── README.md
├── .gitignore
└── .github/workflows/ci.yml
```

## 1) Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 2) Dataset Layout

Prepare the dataset in this format:

```text
dataset/
├── cat/
├── dog/
└── elephant/
```

Each subfolder name becomes a class label.

## 3) Training

```bash
python -m training.train \
  --dataset-dir ./dataset \
  --epochs 10 \
  --batch-size 32 \
  --output-model model.h5 \
  --output-class-map class_indices.json
```

Training script details:
- Builds MobileNetV2 backbone with ImageNet weights.
- Freezes early layers, fine-tunes later layers.
- Uses softmax output for multi-class classification.
- Applies augmentation (rotation, shift, zoom, flip).
- Saves:
  - `model.h5`
  - `class_indices.json`

## 4) Run the Web Server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Open:
- `http://localhost:8000` for web UI.

## 5) API Usage

### Endpoint
`POST /predict`

### cURL Example

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@example_animal.jpg"
```

### Response

```json
{
  "predicted_class": "dog",
  "confidence": 0.9532
}
```

## 6) Testing

```bash
pytest -q
```

## 7) Model Explanation

- **Backbone:** MobileNetV2 pretrained on ImageNet learns generic visual features.
- **Transfer Learning:** Reuses pretrained layers and fine-tunes deeper layers for target animal classes.
- **Classifier Head:** GlobalAveragePooling + Dropout + Dense(softmax) for robust classification.
- **Inference Flow:** Uploaded image -> resize to 224x224 -> MobileNetV2 preprocessing -> prediction -> top class and confidence.

## Notes for Production
- For high throughput, consider loading the model at startup and running under multiple Uvicorn workers.
- Use model versioning and artifact storage (e.g., S3, GCS) for deploy pipelines.
- Add authentication/rate limits if exposing public API.

"""FastAPI app for animal species image classification."""

from __future__ import annotations

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.inference import InferenceError, predict_species
from app.model_loader import ModelLoadError, get_model_and_classes

app = FastAPI(title="Animal Species Classifier", version="1.0.0")
templates = Jinja2Templates(directory="app/templates")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict[str, float | str]:
    """Classify uploaded image into one animal species class."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded image is empty.")

    try:
        model, class_names = get_model_and_classes()
        predicted_class, confidence = predict_species(model, class_names, image_bytes)
    except (ModelLoadError, InferenceError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Unexpected server error.") from exc

    return {
        "predicted_class": predicted_class,
        "confidence": round(confidence, 4),
    }

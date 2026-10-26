import io
import os

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from transformers import AutoImageProcessor, AutoModelForImageClassification

# Folder you saved with model.save_pretrained(...) in Colab (download it from Drive)
MODEL_PATH = os.getenv("MODEL_PATH", "./vitiligo_vit_model")
MAX_FILE_MB = 10
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = AutoImageProcessor.from_pretrained(MODEL_PATH)
model = AutoModelForImageClassification.from_pretrained(MODEL_PATH).to(device).eval()
id2label = model.config.id2label

app = FastAPI(title="Vitiligo ViT Classifier")
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def index():
    return FileResponse("static/index.html")


@app.get("/health")
def health():
    return {"status": "ok", "device": str(device), "classes": list(id2label.values())}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(400, "Please upload a JPG, PNG, WEBP or BMP image.")

    data = await file.read()
    if len(data) > MAX_FILE_MB * 1024 * 1024:
        raise HTTPException(400, f"File too large (max {MAX_FILE_MB} MB).")

    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "Could not read that image file.")

    # Same resize + normalisation as the notebook (224x224, processor mean/std)
    inputs = processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        probs = torch.softmax(model(**inputs).logits, dim=1)[0].cpu().tolist()

    results = sorted(
        [{"label": id2label[i], "probability": p} for i, p in enumerate(probs)],
        key=lambda r: r["probability"],
        reverse=True,
    )
    return {
        "prediction": results[0]["label"],
        "confidence": results[0]["probability"],
        "probabilities": results,
        "width": img.width,
        "height": img.height,
    }

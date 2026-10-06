"""Serve the trained GRU emotion classifier and its web interface."""

from contextlib import asynccontextmanager
from pathlib import Path
import pickle

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR
STATIC_DIR = BASE_DIR / "static"
LABELS = ("sadness", "joy", "love", "anger", "fear", "surprise")
MAX_SEQUENCE_LENGTH = 30


class TextInput(BaseModel):
    text: str = Field(min_length=1, max_length=2000)


class PredictionResponse(BaseModel):
    text: str
    predicted_emotion: str
    confidence: float
    all_probabilities: dict[str, float]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Loading once at startup also makes a missing model fail visibly in deploy logs.
    app.state.model = load_model(MODEL_DIR / "GRU_Model.keras", compile=False)
    with (MODEL_DIR / "GRU_Tokenizer.pkl").open("rb") as file:
        app.state.tokenizer = pickle.load(file)
    yield
    app.state.model = None
    app.state.tokenizer = None


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": app.state.model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict_emotion(payload: TextInput):
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="Enter some text to analyze.")

    # Training used the Keras Tokenizer directly on the original text.
    sequence = app.state.tokenizer.texts_to_sequences([payload.text])
    padded = pad_sequences(
        sequence, maxlen=MAX_SEQUENCE_LENGTH, padding="post", truncating="post"
    )
    probabilities = app.state.model.predict(padded, verbose=0)[0]
    if len(probabilities) != len(LABELS):
        raise HTTPException(status_code=500, detail="Unexpected model output shape.")

    top_index = int(np.argmax(probabilities))
    return PredictionResponse(
        text=payload.text,
        predicted_emotion=LABELS[top_index],
        confidence=float(probabilities[top_index]),
        all_probabilities={
            label: float(score) for label, score in zip(LABELS, probabilities)
        },
    )

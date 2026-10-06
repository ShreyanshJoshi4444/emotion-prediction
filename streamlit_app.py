"""Public Streamlit interface for the trained GRU emotion classifier."""

from pathlib import Path
import pickle

import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


BASE_DIR = Path(__file__).resolve().parent
LABELS = ("sadness", "joy", "love", "anger", "fear", "surprise")
MAX_SEQUENCE_LENGTH = 30
EMOJI = {
    "sadness": "😢",
    "joy": "😊",
    "love": "❤️",
    "anger": "😠",
    "fear": "😨",
    "surprise": "😮",
}


@st.cache_resource(show_spinner="Loading the emotion model...")
def load_assets():
    model = load_model(BASE_DIR / "GRU_Model.keras", compile=False)
    with (BASE_DIR / "GRU_Tokenizer.pkl").open("rb") as file:
        tokenizer = pickle.load(file)
    return model, tokenizer


def predict(text, model, tokenizer):
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(
        sequence, maxlen=MAX_SEQUENCE_LENGTH, padding="post", truncating="post"
    )
    probabilities = np.asarray(model.predict(padded, verbose=0)[0], dtype=float)
    if probabilities.size != len(LABELS):
        raise ValueError("The model returned an unexpected number of emotions.")
    return dict(zip(LABELS, probabilities.tolist()))


st.set_page_config(page_title="EmotiSense", page_icon="💬", layout="centered")
st.title("💬 EmotiSense")
st.caption("Explore the emotion behind your words with a trained GRU classifier.")

with st.form("emotion_form"):
    text = st.text_area(
        "Your text",
        placeholder="For example: I can't wait to see my friends this weekend!",
        height=150,
        max_chars=2000,
    )
    submitted = st.form_submit_button("Analyze emotion", type="primary")

if submitted:
    if not text.strip():
        st.warning("Enter some text to analyze.")
    else:
        try:
            model, tokenizer = load_assets()
            scores = predict(text, model, tokenizer)
        except Exception:
            st.error("The model could not make a prediction. Please try again later.")
            raise

        emotion = max(scores, key=scores.get)
        st.subheader(f"{EMOJI[emotion]} {emotion.title()}")
        st.write(f"Confidence: **{scores[emotion]:.1%}**")
        st.markdown("#### All emotions")
        for label, probability in sorted(scores.items(), key=lambda item: -item[1]):
            st.write(f"{EMOJI[label]} **{label.title()}** — {probability:.1%}")
            st.progress(min(1.0, max(0.0, probability)))

st.divider()
st.caption("Predictions are estimates from a trained model, not a measure of anyone's wellbeing.")

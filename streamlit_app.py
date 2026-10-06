"""Public Streamlit interface for the trained GRU emotion classifier."""

import logging
from pathlib import Path
import pickle

import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


BASE_DIR = Path(__file__).resolve().parent
LABELS = ("sadness", "joy", "love", "anger", "fear", "surprise")
EMOTIONS = {
    "sadness": ("😢", "#7ea9ee"),
    "joy": ("😊", "#f8c96d"),
    "love": ("❤️", "#ef91b7"),
    "anger": ("😠", "#f29286"),
    "fear": ("😨", "#ae9af4"),
    "surprise": ("😮", "#78dacb"),
}
# Evaluation output recorded in the project's training notebook.
MODEL_ACCURACY = {"GRU": .9035, "LSTM": .8875, "BiGRU": .8825, "RNN": .4520}


@st.cache_resource(show_spinner="Loading the emotion model...")
def load_assets():
    model = load_model(BASE_DIR / "GRU_Model.keras", compile=False)
    with (BASE_DIR / "GRU_Tokenizer.pkl").open("rb") as file:
        tokenizer = pickle.load(file)
    return model, tokenizer


def predict(text, model, tokenizer):
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=30, padding="post", truncating="post")
    probabilities = np.asarray(model.predict(padded, verbose=0)[0], dtype=float)
    if probabilities.size != len(LABELS):
        raise ValueError("Unexpected model output shape.")
    return dict(zip(LABELS, probabilities.tolist()))


st.set_page_config(
    page_title="EmotiSense | Emotion prediction",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;600;700;800&display=swap');
:root { color-scheme: dark; }
.stApp { color:#eaf0fa; background:
  radial-gradient(circle at 79% 4%,rgba(126,99,221,.19),transparent 35rem),
  radial-gradient(circle at 5% 37%,rgba(42,173,188,.11),transparent 32rem),#0a101d; }
.block-container { max-width:1120px; padding-top:2.7rem; padding-bottom:4rem; }
html,body,[class*="css"] { font-family:"DM Sans",sans-serif; }
.brand,.hero-title,.stat-value,.result-title,.section-head { font-family:"Manrope",sans-serif; }
.brand { display:flex;align-items:center;gap:.65rem;font-weight:800;color:#f7f9ff;
  font-size:1.2rem;letter-spacing:-.04em; }
.mark { display:inline-grid;place-items:center;width:2.2rem;height:2.2rem;
  border-radius:.75rem;background:linear-gradient(135deg,#80e4d4,#9b8cf7);color:#11192d; }
.eyebrow,.card-label,.result-kicker { text-transform:uppercase;letter-spacing:.16em;
  color:#88dfd3;font-size:.7rem;font-weight:800; }
.eyebrow { margin-top:3.1rem;margin-bottom:.8rem; }
.hero-title { color:#f8faff;font-weight:800;font-size:clamp(2.6rem,5.2vw,4.65rem);
  letter-spacing:-.065em;line-height:1.1;margin:0 0 1rem; }
.hero-title span { background:linear-gradient(95deg,#8be9d5,#a2a6f8,#d3b2f4);
  -webkit-background-clip:text;background-clip:text;color:transparent; }
.hero-copy { color:#aab6cb;font-size:1.04rem;line-height:1.7;
  max-width:44rem;margin-bottom:2.1rem; }
.aside,.result,.info { background:linear-gradient(145deg,#1e2a43,#142036);
  border:1px solid rgba(177,194,232,.17);border-radius:1.35rem;
  box-shadow:0 18px 48px rgba(0,0,0,.13); }
.aside { padding:1.7rem;min-height:20rem; }
.stat-value { font-size:3.4rem;color:#f7f9ff;font-weight:800;letter-spacing:-.08em;
  margin:.8rem 0 .1rem;line-height:1.1; }
.stat-value span { font-size:2rem;color:#8ce2d5; }
.muted,.section-copy { color:#adbad0;font-size:.9rem;line-height:1.6; }
.mini-grid { display:grid;grid-template-columns:1fr 1fr;gap:.7rem;margin-top:1.35rem; }
.mini { background:rgba(255,255,255,.045);border:1px solid rgba(255,255,255,.06);
  border-radius:.85rem;padding:.75rem .9rem; }
.mini small { display:block;color:#a2b1ca;text-transform:uppercase;
  letter-spacing:.1em;font-size:.65rem;font-weight:800; }
.mini strong { display:block;margin-top:.2rem;color:#f6f9ff;font-size:1.1rem; }
[data-testid="stForm"] { background:linear-gradient(145deg,#1e2a43,#142036);
  border:1px solid rgba(177,194,232,.17);border-radius:1.35rem;
  padding:1.3rem 1.4rem 1.15rem;box-shadow:0 18px 48px rgba(0,0,0,.13); }
[data-testid="stTextArea"] textarea { background:#101a2e;color:#f2f6ff;
  border:1px solid #35445f;border-radius:.8rem;font-size:1rem;line-height:1.65; }
[data-testid="stTextArea"] label { color:#eaf0fb;font-weight:700; }
[data-testid="stFormSubmitButton"] button { background:linear-gradient(100deg,#7ce0d0,#ac9df6);
  border:0;color:#102035;border-radius:.75rem;font-weight:800;padding:.55rem 1.25rem; }
[data-testid="stFormSubmitButton"] button:hover { background:#a5eee0;color:#102035;border:0; }
.section-head { color:#f5f7fc;font-weight:800;font-size:1.42rem;
  letter-spacing:-.035em;margin:2.8rem 0 .25rem; }
.result { padding:1.4rem 1.55rem;margin:.75rem 0 1rem;
  background:linear-gradient(110deg,rgba(64,94,123,.65),rgba(58,52,102,.62)); }
.result-title { font-size:2rem;color:#f8faff;font-weight:800;
  letter-spacing:-.055em;margin:.25rem 0; }
.result-note { color:#c7d2e3;font-size:.88rem; }
.prob-row,.model-row { display:flex;align-items:center;gap:.7rem;margin:.7rem 0; }
.prob-name { width:6.7rem;flex:none;color:#e0e9f7;font-size:.9rem;font-weight:600; }
.prob-track,.model-track { flex:1;height:.53rem;border-radius:2rem;
  background:#283750;overflow:hidden; }
.prob-fill,.model-fill { height:100%;border-radius:2rem; }
.prob-value { width:3.5rem;text-align:right;color:#b4c0d4;font-size:.85rem; }
.info { padding:1.35rem 1.5rem;margin-top:.4rem;min-height:17rem; }
.info h3 { color:#f6f8ff;font-size:1.08rem;margin:0 0 .7rem; }
.info p,.info li { color:#b2bfd2;font-size:.88rem;line-height:1.65; }
.info ul { padding-left:1.2rem;margin:.5rem 0; }
.model-row { color:#dce6f6;font-size:.85rem; }
.model-name { width:3.1rem;font-weight:700; }
.model-fill { background:linear-gradient(90deg,#67758e,#8ba1c3); }
.model-fill.active { background:linear-gradient(90deg,#69dacd,#ac9cf8); }
.model-value { width:3.1rem;text-align:right;color:#b1bfd3; }
.fine,.footer { color:#93a4bd;font-size:.78rem;line-height:1.6;margin-top:1rem; }
.footer { margin-top:3rem; }
.info a,.footer a { color:#8de1d5;text-decoration:none; }
@media(max-width:700px) { .block-container { padding:1.2rem 1rem 3rem; }
  .eyebrow { margin-top:1.9rem; }.hero-title { font-size:2.55rem; }
  .aside,.info { min-height:0; } }
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="brand"><span class="mark">✦</span> EmotiSense</div>'
    '<div class="eyebrow">NLP / EMOTION CLASSIFICATION</div>'
    '<h1 class="hero-title">Words carry feelings.<br><span>Let’s find them.</span></h1>'
    '<p class="hero-copy">Enter a sentence or a short passage and explore how a trained '
    'GRU model reads its emotional tone across six classes.</p>',
    unsafe_allow_html=True,
)

input_col, stats_col = st.columns([1.45, 1], gap="large", vertical_alignment="top")
with input_col:
    with st.form("emotion_form"):
        text = st.text_area("Your text",
                            placeholder="I can't wait to see my friends this weekend!",
                            height=175, max_chars=2000)
        submitted = st.form_submit_button("Analyze emotion  →", type="primary")
with stats_col:
    st.markdown(
        '<div class="aside"><div class="card-label">MODEL AT A GLANCE</div>'
        '<div class="stat-value">90.35<span>%</span></div>'
        '<div class="muted">Notebook evaluation accuracy on 2,000 examples.*</div>'
        '<div class="mini-grid">'
        '<div class="mini"><small>Training rows</small><strong>16,000</strong></div>'
        '<div class="mini"><small>Emotions</small><strong>6 classes</strong></div>'
        '<div class="mini"><small>Architecture</small><strong>2 GRU layers</strong></div>'
        '<div class="mini"><small>Input length</small><strong>30 tokens</strong></div>'
        '</div></div>', unsafe_allow_html=True)

if submitted:
    if not text.strip():
        st.warning("Enter some text to analyze.")
        st.session_state.pop("prediction", None)
    else:
        try:
            model, tokenizer = load_assets()
            st.session_state.prediction = predict(text, model, tokenizer)
        except Exception:
            logging.exception("Emotion prediction failed")
            st.session_state.pop("prediction", None)
            st.error("The model could not make a prediction. Please try again later.")

if "prediction" in st.session_state:
    scores = st.session_state.prediction
    emotion = max(scores, key=scores.get)
    emoji, _ = EMOTIONS[emotion]
    st.markdown('<div class="section-head">Your result</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="result"><div class="result-kicker">DOMINANT EMOTION</div>'
        f'<div class="result-title">{emoji} {emotion.title()}</div>'
        f'<div class="result-note">Model score: {scores[emotion]:.1%}. '
        'Scores are relative predictions, not a measure of emotional wellbeing.</div></div>',
        unsafe_allow_html=True)
    st.markdown('<p class="section-copy">The full distribution across all six emotions</p>',
                unsafe_allow_html=True)
    rows = []
    for label, probability in sorted(scores.items(), key=lambda item: -item[1]):
        icon, color = EMOTIONS[label]
        width = min(100, max(0, probability * 100))
        rows.append(
            f'<div class="prob-row"><span class="prob-name">{icon} {label.title()}</span>'
            f'<div class="prob-track"><div class="prob-fill" style="width:{width:.2f}%;'
            f'background:{color}"></div></div><span class="prob-value">{probability:.1%}</span></div>'
        )
    st.markdown("".join(rows), unsafe_allow_html=True)

st.markdown('<div class="section-head">Inside the model</div>'
            '<p class="section-copy">How it was trained and how it compares in the notebook.</p>',
            unsafe_allow_html=True)
method_col, compare_col = st.columns(2, gap="large")
with method_col:
    st.markdown(
        '<div class="info"><h3>How the classifier works</h3>'
        '<p>Text is converted to word IDs, padded or trimmed to 30 tokens, then passed '
        'through an embedding and two GRU layers (128 and 64 units). A six-way softmax '
        'produces the scores shown above.</p>'
        '<ul><li>Tokenizer limit: 7,000 words</li>'
        '<li>Training: class weights, dropout, and early stopping</li>'
        '<li>Dataset: <a href="https://huggingface.co/datasets/dair-ai/emotion" '
        'target="_blank" rel="noopener noreferrer">dair-ai/emotion</a></li></ul></div>',
        unsafe_allow_html=True)
with compare_col:
    bars = []
    for name, accuracy in MODEL_ACCURACY.items():
        active = " active" if name == "GRU" else ""
        bars.append(
            f'<div class="model-row"><span class="model-name">{name}</span>'
            f'<div class="model-track"><div class="model-fill{active}" '
            f'style="width:{accuracy * 100:.2f}%"></div></div>'
            f'<span class="model-value">{accuracy:.1%}</span></div>'
        )
    st.markdown(
        '<div class="info"><h3>Notebook model comparison</h3>' + "".join(bars)
        + '<p class="fine">Accuracy values are from the project training notebook. '
        'The GRU also recorded an evaluation loss of 0.281.</p></div>',
        unsafe_allow_html=True)

st.markdown(
    '<p class="fine">*The 2,000-row test split was also used as validation data during '
    'training. The 90.35% figure is therefore not an independent held-out test '
    'estimate; performance on new writing may differ.</p>'
    '<div class="footer">A portfolio demonstration of NLP emotion classification. '
    'The model predicts sadness, joy, love, anger, fear, or surprise. '
    '<a href="https://github.com/ShreyanshJoshi4444/emotion-prediction" '
    'target="_blank" rel="noopener noreferrer">View source on GitHub ↗</a></div>',
    unsafe_allow_html=True)

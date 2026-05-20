import streamlit as st
import pickle
import numpy as np
import pandas as pd
import scipy.sparse as sp

from utils import clean_text, extract_behavioural_features, hybrid_predict, compute_heuristic_score

st.set_page_config(page_title="Fake Review Detector", page_icon="🔍", layout="centered")

# --- Custom 3D & Glassmorphism CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;600;700&display=swap');

    html, body, p, div, h1, h2, h3, h4, h5, h6, span, label, li, ul, ol, button, input, textarea {
        font-family: 'Space Grotesk', sans-serif;
    }
    
    /* Ensure Streamlit's Material Icons (like _arrow_right_) still render as icons */
    .material-icons, [class^="stIcon"], [class*="icon"], [class*="Icon"], i {
        font-family: 'Material Symbols Rounded', 'Material Icons', sans-serif !important;
    }

    /* Animated 3D Liquid Background */
    .stApp {
        background-color: #050510 !important; /* Very dark background behind everything */
    }

    [data-testid="stAppViewContainer"] {
        background-color: transparent !important;
        overflow-x: hidden;
        z-index: 10; /* Force the UI container to sit above the stApp blobs */
        position: relative;
    }

    /* Big liquid blob on the left (Purple / Blue) */
    .stApp::before {
        content: "";
        position: fixed;
        top: -10%; left: -10%;
        width: 40vw; height: 120vh;
        background: radial-gradient(ellipse at 30% 30%, #a855f7, #3b82f6 50%, transparent 90%);
        border-radius: 40% 60% 70% 30% / 40% 50% 60% 50%;
        animation: liquidBlob 18s ease-in-out infinite alternate;
        filter: drop-shadow(0 0 50px rgba(59, 130, 246, 0.4));
        opacity: 0.7;
        z-index: 0; /* Above stApp background, below stAppViewContainer */
        pointer-events: none;
    }
    
    /* Smaller floating liquid drops (Pink / Deep Purple) */
    .stApp::after {
        content: "";
        position: fixed;
        top: 60%; left: 25%;
        width: 15vw; height: 15vw;
        background: radial-gradient(circle at 25% 25%, #ec4899, #8b5cf6 60%, transparent);
        border-radius: 60% 40% 30% 70% / 60% 30% 70% 40%;
        animation: liquidDrop 12s ease-in-out infinite alternate;
        filter: drop-shadow(0 0 30px rgba(236, 72, 153, 0.6));
        opacity: 0.8;
        z-index: 0;
        pointer-events: none;
        box-shadow: 
            35vw -45vh 0 -5vw #3b82f6, /* Create a third blue drop using box-shadow */
            inset 20px 20px 20px rgba(255,255,255,0.1);
    }

    @keyframes liquidBlob {
        0% {
            border-radius: 40% 60% 70% 30% / 40% 50% 60% 50%;
            transform: translate(0, 0) rotate(0deg) scale(1);
        }
        50% {
            border-radius: 70% 30% 50% 50% / 30% 40% 60% 70%;
            transform: translate(2vw, 2vh) rotate(5deg) scale(1.02);
        }
        100% {
            border-radius: 60% 40% 30% 70% / 50% 60% 40% 60%;
            transform: translate(-2vw, -2vh) rotate(-5deg) scale(0.98);
        }
    }

    @keyframes liquidDrop {
        0% {
            border-radius: 60% 40% 30% 70% / 60% 30% 70% 40%;
            transform: translate(0, 0) rotate(0deg) scale(1);
        }
        50% {
            border-radius: 40% 60% 50% 50% / 40% 50% 70% 30%;
            transform: translate(3vw, -2vh) rotate(15deg) scale(1.05);
        }
        100% {
            border-radius: 40% 60% 70% 30% / 50% 60% 30% 60%;
            transform: translate(-1vw, -5vh) rotate(45deg) scale(1.1);
        }
    }
    
    /* Ensure Streamlit content absolutely sits above the animated background */
    .stApp > header { background: transparent !important; z-index: 999; }
    .main .block-container { z-index: 999; position: relative; }

    /* Premium Title */
    h1 {
        background: linear-gradient(to right, #fbbf24, #f43f5e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700 !important;
        text-shadow: 0 0 20px rgba(244, 63, 94, 0.4);
    }
    
    /* True Glassmorphism Text Areas & Inputs */
    .stTextArea textarea, .stTextInput input {
        background: rgba(20, 25, 40, 0.6) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 16px !important;
        color: white !important;
        box-shadow: inset 0 2px 10px rgba(0, 0, 0, 0.5), 0 8px 32px rgba(0, 0, 0, 0.3) !important;
        backdrop-filter: blur(16px) saturate(180%) !important;
        -webkit-backdrop-filter: blur(16px) saturate(180%) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        font-size: 16px !important;
        padding: 16px !important;
    }
    
    .stTextArea textarea:focus, .stTextInput input:focus {
        border-color: #06b6d4 !important;
        box-shadow: inset 0 2px 10px rgba(0, 0, 0, 0.5), 0 0 20px rgba(6, 182, 212, 0.4) !important;
        transform: translateY(-2px);
    }

    /* 3D Premium Bouncy Button */
    .stButton>button {
        background: linear-gradient(135deg, #06b6d4 0%, #3b82f6 100%) !important;
        border: 1px solid rgba(255,255,255,0.2) !important;
        border-radius: 50px !important;
        color: white !important;
        font-weight: 700 !important;
        font-size: 16px !important;
        text-transform: uppercase !important;
        letter-spacing: 2px !important;
        padding: 12px 24px !important;
        box-shadow: 0 8px 20px rgba(6, 182, 212, 0.4), inset 0 2px 0 rgba(255,255,255,0.3), inset 0 -2px 0 rgba(0,0,0,0.2) !important;
        transform: translateY(0) scale(1) !important;
        transition: all 0.3s cubic-bezier(0.34, 1.56, 0.64, 1) !important;
    }

    .stButton>button:hover {
        transform: translateY(-4px) scale(1.02) !important;
        box-shadow: 0 12px 25px rgba(59, 130, 246, 0.6), inset 0 2px 0 rgba(255,255,255,0.4), inset 0 -2px 0 rgba(0,0,0,0.2) !important;
        border-color: rgba(255,255,255,0.4) !important;
    }
    
    .stButton>button:active {
        transform: translateY(2px) scale(0.98) !important;
        box-shadow: 0 2px 5px rgba(6, 182, 212, 0.4), inset 0 2px 0 rgba(0,0,0,0.1) !important;
    }

    /* Glassmorphism Metric Cards */
    [data-testid="metric-container"] {
        background: rgba(20, 25, 40, 0.6) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 16px !important;
        backdrop-filter: blur(16px) saturate(180%) !important;
        -webkit-backdrop-filter: blur(16px) saturate(180%) !important;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3) !important;
        padding: 15px !important;
        transition: transform 0.3s ease;
    }

    /* Glassmorphism Expanders (applied to container to avoid breaking arrow layout) */
    [data-testid="stExpander"] {
        background: rgba(20, 25, 40, 0.6) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 16px !important;
        backdrop-filter: blur(16px) saturate(180%) !important;
        -webkit-backdrop-filter: blur(16px) saturate(180%) !important;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3) !important;
        overflow: hidden !important;
    }
    
    [data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        border-color: rgba(168, 85, 247, 0.3) !important;
    }

    /* Progress Bar 3D */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #ec4899, #8b5cf6) !important;
        box-shadow: 0 0 15px rgba(236, 72, 153, 0.5) !important;
        border-radius: 10px !important;
    }
    
    /* Typography Overrides */
    p, label {
        color: #e2e8f0 !important;
    }
    
    [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-weight: 700 !important;
        text-shadow: 0 0 10px rgba(255,255,255,0.2);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    try:
        with open("model/model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("model/tfidf.pkl", "rb") as f:
            tfidf = pickle.load(f)
        with open("model/scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        with open("model/meta.pkl", "rb") as f:
            meta = pickle.load(f)
        return model, tfidf, scaler, meta
    except FileNotFoundError as e:
        st.error(f"Model files not found: {e}\n\nPlease run `python train.py` first.")
        st.stop()


model, tfidf, scaler, meta = load_model()


def predict(text, rating=3.0):
    """Predict whether a review is fake or real."""
    pred, fake_prob, real_prob = hybrid_predict(text, rating, model, tfidf, scaler)
    return pred, fake_prob, real_prob


def get_signals(text, rating=3.0):
    """Detect human-readable warning signals in a review."""
    _, signals = compute_heuristic_score(text, rating)
    return signals


# ── UI ────────────────────────────────────────────────────────
st.title("🔍 Fake Review Detector")
st.caption(f"Trained on 40,000 real Amazon reviews · {meta['model_name']}")

with st.expander("Model performance"):
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Best model", meta["model_name"])
    col2.metric("Accuracy", f"{meta['accuracy']:.1%}")
    col3.metric("F1-score", f"{meta['f1']:.1%}")
    col4.metric("AUC-ROC", f"{meta.get('auc', 0):.1%}")

    st.markdown("**All models compared:**")
    rows = []
    for name, res in meta["all_results"].items():
        rows.append({
            "Model": name,
            "Accuracy": f"{res['accuracy']:.1%}",
            "F1": f"{res['f1']:.1%}",
            "AUC-ROC": f"{res.get('auc', 0):.1%}",
            "CV F1": f"{res.get('cv_f1_mean', 0):.4f} ± {res.get('cv_f1_std', 0):.4f}",
        })
    st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)

st.divider()

review = st.text_area(
    "Paste your review here:",
    height=150,
    placeholder="Type or paste a product review here...",
)
rating = st.slider("Star rating of this review:", 1.0, 5.0, 3.0, step=1.0)

if st.button("Analyze Review", type="primary", width='stretch'):
    # Strip surrounding quotes (users often paste "review" from ChatGPT)
    review_clean = review.strip().strip('""\u201c\u201d\u2018\u2019\'')
    if not review_clean.strip():
        st.warning("Please enter a review first.")
    elif len(review_clean.split()) < 3:
        st.warning("Too short — enter at least a few words.")
    else:
        pred, fake_prob, real_prob = predict(review_clean, rating)
        st.divider()

        if pred == 1:
            st.error("## ⚠️ FAKE review detected")
        else:
            st.success("## ✅ Looks like a REAL review")

        col1, col2 = st.columns(2)
        col1.metric("Real probability", f"{real_prob:.1%}")
        col2.metric("Fake probability", f"{fake_prob:.1%}")
        st.progress(float(fake_prob), text=f"Fake score: {fake_prob:.1%}")

        signals = get_signals(review_clean, rating)
        if signals:
            st.markdown("**Signals detected:**")
            for s in signals:
                st.markdown(f"- ⚡ {s}")

        with st.expander("Review stats"):
            words = review_clean.split()
            st.markdown(f"- Words: **{len(words)}**")
            st.markdown(f"- Sentences: **{max(len([s for s in review_clean.split('.') if s.strip()]), 1)}**")
            st.markdown(f"- Exclamation marks: **{review_clean.count('!')}**")
            caps = sum(1 for c in review_clean if c.isupper())
            st.markdown(f"- CAPS ratio: **{caps / max(len(review_clean), 1):.1%}**")
            st.markdown(
                f"- Unique word ratio: **{len(set(review_clean.lower().split())) / max(len(words), 1):.1%}**"
            )

st.divider()
with st.expander("Batch analyze — one review per line"):
    batch = st.text_area("Paste multiple reviews here:", height=120)
    batch_rating = st.slider("Star rating for batch:", 1.0, 5.0, 3.0, step=1.0, key="batch_rating")
    if st.button("Analyze all"):
        lines = [l.strip() for l in batch.strip().split("\n") if l.strip()]
        if lines:
            out = []
            for line in lines:
                p, fp, rp = predict(line, batch_rating)
                out.append(
                    {
                        "Review": line[:70] + ("..." if len(line) > 70 else ""),
                        "Result": "⚠️ FAKE" if p == 1 else "✅ REAL",
                        "Fake %": f"{fp:.1%}",
                        "Real %": f"{rp:.1%}",
                    }
                )
            st.dataframe(pd.DataFrame(out), width='stretch', hide_index=True)

st.caption(f"Fake Review Detector · {meta['model_name']} · Python, Scikit-learn & Streamlit")
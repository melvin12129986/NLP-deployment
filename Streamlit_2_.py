import io
import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

try:
    import joblib
except ImportError:
    st.error("joblib not installed. Run: pip install joblib")
    st.stop()

from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ── Custom CSS ────────────────────────────────────────────────────────────────
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Sora', sans-serif !important; color: #000 !important; }
body, html, div, span, p, label, a, li, h1, h2, h3, h4, h5, h6, button, textarea, input {
    color: #000 !important;
}

/* ── Page background ── */
.stApp { background: #F7F6F2; }

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem !important; padding-bottom: 3rem !important; max-width: 860px !important; }

/* ── Top header bar ── */
.app-header {
    display: flex; align-items: center; gap: 14px;
    padding: 1.5rem 0 1.25rem; margin-bottom: 0.5rem;
    border-bottom: 1px solid #D3D1C7;
}
.logo-mark {
    width: 42px; height: 42px; background: #1D9E75; border-radius: 11px;
    display: flex; align-items: center; justify-content: center; flex-shrink: 0;
    font-size: 22px;
}
.header-text h1 { font-size: 22px; font-weight: 600; color: #000 !important; margin: 0; }
.header-text p  { font-size: 13px; color: #000 !important; margin: 0; }

/* ── Stat cards ── */
div[data-testid="metric-container"] {
    background: #EEECEA; border-radius: 10px !important;
    padding: 0.9rem 1.1rem !important; border: none !important;
}
div[data-testid="metric-container"] label { font-size: 11px !important; letter-spacing: 0.06em; text-transform: uppercase; color: #000 !important; }
div[data-testid="metric-container"] [data-testid="metric-value"] {
    font-size: clamp(14px, 2.4vw, 22px) !important;
    font-weight: 600 !important;
    color: #000 !important;
    white-space: normal !important;
    overflow: visible !important;
    word-break: break-word !important;
    line-height: 1.25 !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #EEECEA; border-radius: 10px; padding: 5px; gap: 4px; border-bottom: none !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important; border-radius: 7px !important;
    font-size: 13px !important; font-weight: 500; color: #000 !important;
    font-family: 'Sora', sans-serif !important; padding: 7px 20px !important;
    border: none !important;
}
.stTabs [aria-selected="true"] {
    background: #fff !important; color: #000 !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.10) !important;
}
.stTabs [data-baseweb="tab-highlight"] { display: none; }
.stTabs [data-baseweb="tab-border"]    { display: none; }

/* ── Text area ── */
textarea {
    border: 1px solid #C4C2B8 !important; border-radius: 10px !important;
    font-family: 'Sora', sans-serif !important; font-size: 14px !important;
    background: #fff !important; color: #2C2C2A !important;
    transition: border-color 0.15s, box-shadow 0.15s;
}
textarea:focus { border-color: #1D9E75 !important; box-shadow: 0 0 0 3px rgba(29,158,117,0.12) !important; }

/* ── Text input (search) ── */
div[data-baseweb="input"] input {
    border: 1px solid #C4C2B8 !important; border-radius: 10px !important;
    font-family: 'Sora', sans-serif !important; font-size: 13px !important;
    background: #fff !important; padding-left: 16px !important;
}
div[data-baseweb="input"] input:focus { border-color: #1D9E75 !important; box-shadow: 0 0 0 3px rgba(29,158,117,0.12) !important; }

/* ── Primary button ── */
.stButton > button {
    background: #1D9E75 !important; color: #fff !important;
    border: none !important; border-radius: 10px !important;
    font-family: 'Sora', sans-serif !important; font-size: 14px !important;
    font-weight: 600 !important; padding: 0.65rem 1.5rem !important;
    width: 100% !important; transition: background 0.15s, transform 0.1s !important;
}
.stButton > button:hover  { background: #085041 !important; }
.stButton > button:active { transform: scale(0.98) !important; }

/* ── Progress bar ── */
.stProgress > div > div { border-radius: 99px !important; background: #E1F5EE !important; }
.stProgress > div > div > div { border-radius: 99px !important; background: #1D9E75 !important; }

/* ── Result cards ── */
.result-card {
    background: #fff; border-radius: 12px; border: 1px solid #D3D1C7;
    padding: 1rem 1.25rem; margin-bottom: 10px;
    transition: border-color 0.15s;
}
.result-card.top { border-color: #1D9E75; border-width: 1.5px; }
.rank-badge {
    display: inline-flex; align-items: center; justify-content: center;
    width: 22px; height: 22px; border-radius: 50%; font-size: 11px; font-weight: 600;
}
.rank-1 { background: #1D9E75; color: #fff; }
.rank-n { background: #E1F5EE; color: #085041; }

/* ── Disclaimer banner ── */
.disclaimer {
    background: #F1EFE8; border-radius: 10px; padding: 10px 16px;
    font-size: 12.5px; color: #5F5E5A; line-height: 1.6; margin-top: 1.25rem;
    display: flex; gap: 8px; align-items: flex-start;
}

/* ── Disease pills ── */
.disease-pill {
    background: #EEECEA; border-radius: 8px; padding: 7px 12px;
    font-size: 12.5px; color: #444441;
    border: 1px solid transparent; margin-bottom: 4px;
    display: inline-block; width: 100%;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] { background: #F1EFE8 !important; }
section[data-testid="stSidebar"] .block-container { padding-top: 1.5rem !important; }

/* ── Expander ── */
details { background: #fff !important; border-radius: 10px !important; border: 1px solid #D3D1C7 !important; }

/* ── Dataframe ── */
.stDataFrame { border-radius: 10px !important; overflow: hidden !important; border: 1px solid #D3D1C7 !important; }

/* ── Info / warning boxes ── */
.stAlert { border-radius: 10px !important; font-family: 'Sora', sans-serif !important; }
</style>
"""

# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_dataset_from_path(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, header=None, names=["id", "disease", "symptoms"])
    df["disease"]   = df["disease"].astype(str).str.strip()
    df["symptoms"]  = df["symptoms"].astype(str).str.strip()
    return df

@st.cache_data
def load_dataset_from_bytes(file_bytes: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(file_bytes), header=None, names=["id", "disease", "symptoms"])
    df["disease"]   = df["disease"].astype(str).str.strip()
    df["symptoms"]  = df["symptoms"].astype(str).str.strip()
    return df

@st.cache_resource
def load_model_and_vectorizer(model_path: str, vectorizer_path: str):
    try:
        return joblib.load(model_path), joblib.load(vectorizer_path)
    except FileNotFoundError as e:
        st.error(f"Model files not found: {e}")
        return None, None

@st.cache_resource
def detect_svm_model_paths():
    base_dir = Path(__file__).resolve().parent
    model_path = base_dir / "svm_model_tfidf.pkl"
    vectorizer_path = base_dir / "tfidf_vectorizer.pkl"
    if model_path.exists() and vectorizer_path.exists():
        return str(model_path), str(vectorizer_path)
    return None, None

@st.cache_resource
def get_text_cleaner():
    try:
        import nltk
        for pkg in ('stopwords', 'wordnet', 'averaged_perceptron_tagger_eng'):
            nltk.download(pkg, quiet=True)
        return set(stopwords.words('english')), WordNetLemmatizer()
    except Exception as e:
        st.error(f"Error loading NLTK resources: {e}")
        return None, None

# ── Medical normalization (MCN layer) ──────────────────────────────────────────
medical_normalization = {

    # Fever
    "high temperature": "fever",
    "body temperature": "fever",
    "hot body": "fever",
    "mild fever": "fever",
    "high fever": "fever",

    # Breathing
    "difficulty breathing": "dyspnea",
    "shortness of breath": "dyspnea",
    "trouble breathing": "dyspnea",
    "can't breathe": "dyspnea",
    "breathlessness": "dyspnea",

    # Chest
    "pain in chest": "chest pain",
    "chest hurts": "chest pain",
    "tight chest": "chest pain",

    # Stomach
    "stomach ache": "abdominal pain",
    "belly pain": "abdominal pain",
    "pain in stomach": "abdominal pain",

    # Vomit / nausea
    "throwing up": "vomiting",
    "feel like vomiting": "nausea",
    "feeling sick": "nausea",

    # Head
    "head hurts": "headache",
    "migraine pain": "headache",

    # Skin
    "skin rash": "rash",
    "red spots": "rash",
    "itchy skin": "itching",

    # Weakness
    "feeling weak": "fatigue",
    "very tired": "fatigue",
    "low energy": "fatigue",

    # Blood pressure / sugar
    "high blood pressure": "hypertension",
    "high blood sugar": "diabetes",

    # Common abbreviations
    "bp": "blood pressure",
    "sob": "dyspnea",
    "mi": "myocardial infarction"
}


def normalize_medical_text(text):

    text = str(text).lower()

    # Replace medical phrases
    for phrase, normalized in medical_normalization.items():

        text = re.sub(
            r'\b' + re.escape(phrase) + r'\b',
            normalized,
            text
        )

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# ── NLP helpers ───────────────────────────────────────────────────────────────
def get_wordnet_pos(tag):
    if tag.startswith('J'): return 'a'
    if tag.startswith('V'): return 'v'
    if tag.startswith('R'): return 'r'
    return 'n'

def clean_text(text: str, stop_words, stemmer) -> str:
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', ' ', text)
    contractions = {
        "i'm":"i am","i've":"i have","i'll":"i will","i'd":"i would",
        "you're":"you are","you've":"you have","you'll":"you will","you'd":"you would",
        "he's":"he is","she's":"she is","it's":"it is","we're":"we are","we've":"we have",
        "we'll":"we will","they're":"they are","they've":"they have","they'll":"they will",
        "that's":"that is","there's":"there is","can't":"cannot","won't":"will not",
        "don't":"do not","didn't":"did not","isn't":"is not","aren't":"are not",
        "wasn't":"was not","weren't":"were not","shouldn't":"should not",
        "couldn't":"could not","wouldn't":"would not","haven't":"have not",
        "hasn't":"has not","hadn't":"had not",
    }
    for c, r in contractions.items():
        text = re.sub(r'\b' + re.escape(c) + r'\b', r, text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = [w for w in text.split() if w not in stop_words]
    tagged = pos_tag(words)
    return ' '.join(stemmer.lemmatize(w, get_wordnet_pos(t)) for w, t in tagged)

def predict_disease(input_text, svm_model, vectorizer, stop_words, stemmer):
    if not input_text.strip():
        return None
    normalized_input = normalize_medical_text(input_text)
    cleaned = clean_text(normalized_input, stop_words, stemmer)
    if not cleaned:
        return None
    try:
        vec    = vectorizer.transform([cleaned]).toarray()
        scores = svm_model.decision_function(vec)[0]
        top3   = np.argsort(scores)[::-1][:3]
        return [
            {"disease": svm_model.classes_[i], "confidence": float(1 / (1 + np.exp(-scores[i])))}
            for i in top3
        ]
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

# ── Result card HTML ──────────────────────────────────────────────────────────
def render_result_card(rank: int, disease: str, confidence: float):
    pct         = confidence * 100
    badge_cls   = "rank-1" if rank == 1 else "rank-n"
    card_cls    = "result-card top" if rank == 1 else "result-card"
    bar_colors  = ["#1D9E75", "#5DCAA5", "#9FE1CB"]
    bar_color   = bar_colors[rank - 1]

    st.markdown(f"""
    <div class="{card_cls}">
      <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:8px;">
        <span class="rank-badge {badge_cls}">{rank}</span>
        <span style="flex:1;margin:0 10px;font-size:14px;font-weight:600;color:#2C2C2A;">{disease}</span>
        <span style="font-size:13px;font-weight:600;color:#1D9E75;font-family:'DM Mono',monospace;">{pct:.1f}%</span>
      </div>
      <div style="height:5px;background:#E1F5EE;border-radius:99px;overflow:hidden;">
        <div style="width:{pct}%;height:100%;background:{bar_color};border-radius:99px;transition:width 0.6s ease;"></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    st.set_page_config(page_title="Symptom Checker", page_icon="🩺", layout="wide")
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # Header
    st.markdown("""
    <div class="app-header">
      <div class="logo-mark">🩺</div>
      <div class="header-text">
        <h1>Symptom Checker</h1>
        <p>Describe how you feel — the model will match your symptoms to known conditions.</p>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Load dataset ──
    csv_path = Path(__file__).resolve().parent / "Symptom2Disease.csv"
    df = None
    if not csv_path.exists():
        st.error("Dataset file `Symptom2Disease.csv` is missing.")
        uploaded = st.file_uploader("Upload Symptom2Disease.csv", type=["csv"])
        if uploaded:
            df = load_dataset_from_bytes(uploaded.getvalue())
        else:
            st.info("Upload the CSV file to continue.")
            st.stop()
    else:
        df = load_dataset_from_path(str(csv_path))

    # ── Load model ──
    svm_model_path, vectorizer_path = detect_svm_model_paths()
    model_label = "Unknown"
    if svm_model_path and vectorizer_path:
        svm_model, vectorizer = load_model_and_vectorizer(svm_model_path, vectorizer_path)
        model_label = "SVM + TF-IDF"
    else:
        st.error(
            "Required model files are missing. Add `svm_model_tfidf.pkl` and `tfidf_vectorizer.pkl` to the app folder."
        )
        st.stop()

    if not svm_model or not vectorizer:
        st.stop()

    stop_words, stemmer = get_text_cleaner()
    if stop_words is None:
        st.stop()

    diseases      = sorted(df["disease"].unique())
    total_samples = len(df)

    # ── Sidebar ──
    with st.sidebar:
        st.markdown("### 📊 Dataset overview")
        st.metric("Known diseases", len(diseases))
        st.metric("Symptom examples", total_samples)
        st.markdown("---")
        st.caption("This app uses SVM + TF-IDF to classify diseases from symptom text. Demo purposes only — not medical advice.")

    # ── Stat strip ──
    c1, c2, c3 = st.columns(3)
    c1.metric("Known diseases",    len(diseases))
    c2.metric("Symptom examples",  total_samples)
    c3.metric("Model",             model_label)

    st.markdown("<div style='margin-bottom:1.25rem;'></div>", unsafe_allow_html=True)

    # ── Tabs ──
    tab1, tab2 = st.tabs(["🔍  Check symptoms", "📋  Browse diseases"])

    # ─────────────────────── TAB 1 ───────────────────────
    with tab1:
        st.markdown("##### Describe your symptoms")

        COMMON_CHIPS = ["Fever", "Cough", "Headache", "Fatigue", "Nausea",
                        "Sore throat", "Chest pain", "Shortness of breath", "Dizziness", "Rash"]

        # Quick-add chips
        st.markdown("<div style='font-size:12px;color:#888780;margin-bottom:6px;'>Common symptoms — click to add:</div>", unsafe_allow_html=True)
        for chip_row in [COMMON_CHIPS[i:i+5] for i in range(0, len(COMMON_CHIPS), 5)]:
            cols = st.columns(len(chip_row))
            for idx, chip in enumerate(chip_row):
                if cols[idx].button(chip, key=f"chip_{chip}"):
                    current = st.session_state.get("symptom_text", "")
                    st.session_state["symptom_text"] = (current + (", " if current else "") + chip.lower())

        user_input = st.text_area(
            "Your symptoms",
            height=130,
            placeholder="e.g. fever, cough, headache, nausea, muscle pain...",
            key="symptom_text",
            label_visibility="collapsed",
        )

        predict_clicked = st.button("✦  Analyze symptoms", use_container_width=True)

        if predict_clicked:
            if not user_input.strip():
                st.warning("Please describe your symptoms first.")
            else:
                with st.spinner("Analyzing with ML model…"):
                    results = predict_disease(user_input, svm_model, vectorizer, stop_words, stemmer)

                if results is None:
                    st.error("Could not process input. Try using clearer symptom descriptions.")
                else:
                    st.markdown("---")
                    st.markdown("**Top 3 predicted conditions**")
                    for i, pred in enumerate(results, 1):
                        render_result_card(i, pred["disease"], pred["confidence"])

                    st.markdown("<div style='margin-top:1.25rem;'></div>", unsafe_allow_html=True)
                    best = results[0]["disease"]
                    with st.expander(f"Sample symptoms for **{best}**"):
                        examples = df[df["disease"] == best]["symptoms"].head(5).tolist()
                        for ex in examples:
                            st.write(f"• {ex}")

        st.markdown("""
        <div class="disclaimer">
          ℹ️&nbsp; This app is for educational/demo purposes only and is not a substitute for
          professional medical diagnosis or advice. Always consult a qualified healthcare provider.
        </div>
        """, unsafe_allow_html=True)

    # ─────────────────────── TAB 2 ───────────────────────
    with tab2:
        st.markdown("##### Browse known conditions")
        search = st.text_input("🔍  Search diseases", placeholder="Type a name or keyword…", label_visibility="collapsed")
        filtered = [d for d in diseases if search.lower() in d.lower()] if search else diseases

        st.markdown(f"<div style='font-size:12px;color:#888780;margin-bottom:10px;'>{len(filtered)} conditions found</div>", unsafe_allow_html=True)

        left, right = st.columns([3, 1])
        with left:
            st.dataframe(pd.DataFrame({"Disease": filtered}), use_container_width=True, hide_index=True)
        with right:
            st.markdown("**Quick view**")
            for d in filtered[:12]:
                st.markdown(f"<div class='disease-pill'>{d}</div>", unsafe_allow_html=True)

        with st.expander("Show sample symptom descriptions"):
            for disease in filtered[:20]:
                samples = df[df["disease"] == disease]["symptoms"].head(3).tolist()
                if samples:
                    st.markdown(f"**{disease}**")
                    for s in samples:
                        st.write(f"• {s}")


if __name__ == "__main__":
    main()

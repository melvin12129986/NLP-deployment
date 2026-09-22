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
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { 
    font-family: 'Plus Jakarta Sans', sans-serif !important; 
}

/* ── Dark Theme Background ── */
.stApp { background: #0f172a; color: #f8fafc; }

/* ── Hide Streamlit Chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem !important; padding-bottom: 3rem !important; max-width: 960px !important; }

/* ── Header Bar ── */
.app-header {
    display: flex; align-items: center; gap: 16px;
    padding: 1.25rem 1.5rem; margin-bottom: 1.5rem;
    background: #1e293b; border-radius: 16px; border: 1px solid #334155;
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);
}
.logo-mark {
    width: 48px; height: 48px; background: linear-gradient(135deg, #059669 0%, #10b981 100%); 
    border-radius: 12px; display: flex; align-items: center; justify-content: center; flex-shrink: 0;
    font-size: 24px; box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
}
.header-text h1 { font-size: 20px; font-weight: 700; color: #f8fafc !important; margin: 0; }
.header-text p  { font-size: 13px; color: #94a3b8 !important; margin: 2px 0 0; }

/* ── Stat Cards ── */
div[data-testid="metric-container"] {
    background: #1e293b !important; border-radius: 12px !important;
    padding: 0.85rem 1rem !important; border: 1px solid #334155 !important;
}
div[data-testid="metric-container"] label { font-size: 11px !important; text-transform: uppercase; color: #94a3b8 !important; }
div[data-testid="metric-container"] [data-testid="metric-value"] {
    font-size: 15px !important; font-weight: 600 !important; color: #38bdf8 !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #1e293b; border-radius: 12px; padding: 6px; gap: 6px; border-bottom: none !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important; border-radius: 8px !important;
    font-size: 13px !important; font-weight: 500; color: #94a3b8 !important;
    padding: 8px 18px !important; border: none !important;
}
.stTabs [aria-selected="true"] {
    background: #334155 !important; color: #f8fafc !important;
}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }

/* ── Inputs ── */
textarea, div[data-baseweb="input"] input {
    border: 1px solid #334155 !important; border-radius: 12px !important;
    background: #1e293b !important; color: #f8fafc !important;
}
textarea:focus, div[data-baseweb="input"] input:focus {
    border-color: #10b981 !important; box-shadow: 0 0 0 2px rgba(16, 185, 129, 0.2) !important;
}

/* ── Action Button ── */
.stButton > button {
    background: linear-gradient(135deg, #059669 0%, #10b981 100%) !important; color: #fff !important;
    border: none !important; border-radius: 12px !important;
    font-weight: 600 !important; padding: 0.75rem 1.5rem !important;
    width: 100% !important; box-shadow: 0 4px 12px rgba(16, 185, 129, 0.25) !important;
}
.stButton > button:hover { background: #047857 !important; }

/* ── Result Cards ── */
.result-card {
    background: #1e293b; border-radius: 12px; border: 1px solid #334155;
    padding: 1rem 1.25rem; margin-bottom: 12px;
}
.result-card.top { border-color: #10b981; background: linear-gradient(135deg, #1e293b 0%, #064e3b 100%); }
.rank-badge {
    display: inline-flex; align-items: center; justify-content: center;
    width: 24px; height: 24px; border-radius: 50%; font-size: 12px; font-weight: 700;
}
.rank-1 { background: #10b981; color: #0f172a; }
.rank-n { background: #334155; color: #94a3b8; }

/* ── Disclaimer Banner ── */
.disclaimer {
    background: #1e293b; border-radius: 10px; padding: 12px 16px;
    font-size: 12px; color: #94a3b8; line-height: 1.5; margin-top: 1.5rem;
    border-left: 4px solid #f59e0b;
}

section[data-testid="stSidebar"] { background: #1e293b !important; }
</style>
"""

# ── Data Loading Functions ───────────────────────────────────────────────────
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

# ── Medical Normalization (MCN Layer) ─────────────────────────────────────────
medical_normalization = {
    "high temperature": "fever", "body temperature": "fever", "hot body": "fever",
    "mild fever": "fever", "high fever": "fever", "difficulty breathing": "dyspnea",
    "shortness of breath": "dyspnea", "trouble breathing": "dyspnea", "can't breathe": "dyspnea",
    "breathlessness": "dyspnea", "pain in chest": "chest pain", "chest hurts": "chest pain",
    "tight chest": "chest pain", "stomach ache": "abdominal pain", "belly pain": "abdominal pain",
    "pain in stomach": "abdominal pain", "throwing up": "vomiting", "feel like vomiting": "nausea",
    "feeling sick": "nausea", "head hurts": "headache", "migraine pain": "headache",
    "skin rash": "rash", "red spots": "rash", "itchy skin": "itching", "feeling weak": "fatigue",
    "very tired": "fatigue", "low energy": "fatigue", "high blood pressure": "hypertension",
    "high blood sugar": "diabetes", "bp": "blood pressure", "sob": "dyspnea", "mi": "myocardial infarction"
}

def normalize_medical_text(text):
    text = str(text).lower()
    for phrase, normalized in medical_normalization.items():
        text = re.sub(r'\b' + re.escape(phrase) + r'\b', normalized, text)
    return re.sub(r'\s+', ' ', text).strip()

# ── NLP & Inference Helpers ───────────────────────────────────────────────────
def get_wordnet_pos(tag):
    if tag.startswith('J'): return 'a'
    if tag.startswith('V'): return 'v'
    if tag.startswith('R'): return 'r'
    return 'n'

def clean_text(text: str, stop_words, stemmer) -> str:
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', ' ', text)
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
        vec = vectorizer.transform([cleaned]).toarray()
        scores = svm_model.decision_function(vec)[0]
        top3 = np.argsort(scores)[::-1][:3]
        return [
            {"disease": svm_model.classes_[i], "confidence": float(1 / (1 + np.exp(-scores[i])))}
            for i in top3
        ]
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

def render_result_card(rank: int, disease: str, confidence: float):
    pct = confidence * 100
    badge_cls = "rank-1" if rank == 1 else "rank-n"
    card_cls = "result-card top" if rank == 1 else "result-card"
    bar_colors = ["#10b981", "#38bdf8", "#818cf8"]
    bar_color = bar_colors[rank - 1]

    st.markdown(f"""
    <div class="{card_cls}">
      <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:8px;">
        <span class="rank-badge {badge_cls}">{rank}</span>
        <span style="flex:1;margin:0 12px;font-size:15px;font-weight:600;color:#f8fafc;">{disease}</span>
        <span style="font-size:14px;font-weight:600;color:#38bdf8;font-family:'JetBrains Mono',monospace;">{pct:.1f}%</span>
      </div>
      <div style="height:6px;background:#334155;border-radius:99px;overflow:hidden;">
        <div style="width:{pct}%;height:100%;background:{bar_color};border-radius:99px;"></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    st.set_page_config(page_title="AI Health Assistant", page_icon="🩺", layout="wide")
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # Header
    st.markdown("""
    <div class="app-header">
      <div class="logo-mark">🩺</div>
      <div class="header-text">
        <h1>AI Symptom Diagnostic Assistant</h1>
        <p>Describe your physical symptoms to compute preliminary machine learning predictions.</p>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Dataset & Model Setup
    csv_path = Path(__file__).resolve().parent / "Symptom2Disease.csv"
    if not csv_path.exists():
        st.error("Dataset `Symptom2Disease.csv` missing.")
        st.stop()
    df = load_dataset_from_path(str(csv_path))

    svm_model_path, vectorizer_path = detect_svm_model_paths()
    if not svm_model_path or not vectorizer_path:
        st.error("Required model .pkl files missing.")
        st.stop()

    svm_model, vectorizer = load_model_and_vectorizer(svm_model_path, vectorizer_path)
    stop_words, stemmer = get_text_cleaner()

    diseases = sorted(df["disease"].unique())

    # Stat Strip
    c1, c2, c3 = st.columns(3)
    c1.metric("Known Conditions", len(diseases))
    c2.metric("Dataset Samples", len(df))
    c3.metric("ML Engine", "SVM + TF-IDF")

    st.markdown("<div style='margin-bottom:1.5rem;'></div>", unsafe_allow_html=True)

    # Tabs
    tab1, tab2 = st.tabs(["🔍 Check Symptoms", "📋 Browse Database"])

    with tab1:
        st.subheader("Describe Your Symptoms")
        
        COMMON_CHIPS = ["Fever", "Cough", "Headache", "Fatigue", "Nausea",
                        "Sore throat", "Chest pain", "Shortness of breath", "Dizziness", "Rash"]

        st.caption("Quick-add common symptoms:")
        chip_cols = st.columns(5)
        for idx, chip in enumerate(COMMON_CHIPS):
            if chip_cols[idx % 5].button(chip, key=f"chip_{chip}"):
                curr = st.session_state.get("symptom_input", "")
                st.session_state["symptom_input"] = (curr + (", " if curr else "") + chip.lower())

        user_input = st.text_area(
            "Symptoms Input",
            height=120,
            placeholder="e.g. High fever, joint pain, skin rash...",
            key="symptom_input",
            label_visibility="collapsed"
        )

        if st.button("✦ Analyze Symptoms", use_container_width=True):
            if not user_input.strip():
                st.warning("Please input symptoms before running analysis.")
            else:
                with st.spinner("Analyzing text embeddings..."):
                    results = predict_disease(user_input, svm_model, vectorizer, stop_words, stemmer)

                if results:
                    st.markdown("### Clinical Predictions")
                    for i, pred in enumerate(results, 1):
                        render_result_card(i, pred["disease"], pred["confidence"])
                    
                    best = results[0]["disease"]
                    with st.expander(f"Reference symptoms for **{best}**"):
                        examples = df[df["disease"] == best]["symptoms"].head(4).tolist()
                        for ex in examples:
                            st.write(f"• {ex}")

        st.markdown("""
        <div class="disclaimer">
          ⚠️ <strong>Medical Disclaimer:</strong> This application utilizes Machine Learning models for demo purposes only. 
          It is not intended to provide clinical diagnoses. Always seek advice from a medical professional.
        </div>
        """, unsafe_allow_html=True)

    with tab2:
        search = st.text_input("Search conditions...", placeholder="Type condition name...")
        filtered = [d for d in diseases if search.lower() in d.lower()] if search else diseases
        
        st.dataframe(pd.DataFrame({"Disease Name": filtered}), use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()
import io
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

try:
    import joblib
except ImportError:
    st.error("joblib not installed. Please install it with: pip install joblib")
    st.stop()

from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


@st.cache_data
def load_dataset_from_path(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, header=None, names=["id", "disease", "symptoms"])
    df["disease"] = df["disease"].astype(str).str.strip()
    df["symptoms"] = df["symptoms"].astype(str).str.strip()
    return df


@st.cache_data
def load_dataset_from_bytes(file_bytes: bytes) -> pd.DataFrame:
    buffer = io.BytesIO(file_bytes)
    df = pd.read_csv(buffer, header=None, names=["id", "disease", "symptoms"])
    df["disease"] = df["disease"].astype(str).str.strip()
    df["symptoms"] = df["symptoms"].astype(str).str.strip()
    return df


@st.cache_resource
def load_svm_model_and_vectorizer(model_path: str, vectorizer_path: str):
    """Load the SVM model and TF-IDF vectorizer."""
    try:
        svm_model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        return svm_model, vectorizer
    except FileNotFoundError as e:
        st.error(f"Model files not found: {e}")
        return None, None


def get_wordnet_pos(treebank_tag):
    """Map POS tags to WordNet POS tags."""
    if treebank_tag.startswith('J'):
        return 'a'  # adjective
    elif treebank_tag.startswith('V'):
        return 'v'  # verb
    elif treebank_tag.startswith('N'):
        return 'n'  # noun
    elif treebank_tag.startswith('R'):
        return 'r'  # adverb
    else:
        return 'n'  # default to noun


@st.cache_resource
def get_text_cleaner():
    """Initialize and cache text cleaning resources."""
    try:
        import nltk
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('averaged_perceptron_tagger_eng', quiet=True)
        stop_words = set(stopwords.words('english'))
        stemmer = WordNetLemmatizer()
        return stop_words, stemmer
    except Exception as e:
        st.error(f"Error loading NLTK resources: {e}")
        return None, None


def clean_text(text: str, stop_words, stemmer) -> str:
    """Clean and preprocess text."""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', ' ', text)

    contractions = {
        "i'm": "i am", "i've": "i have", "i'll": "i will", "i'd": "i would",
        "you're": "you are", "you've": "you have", "you'll": "you will", "you'd": "you would",
        "he's": "he is", "she's": "she is", "it's": "it is",
        "we're": "we are", "we've": "we have", "we'll": "we will",
        "they're": "they are", "they've": "they have", "they'll": "they will",
        "that's": "that is", "there's": "there is",
        "can't": "cannot", "won't": "will not", "don't": "do not", "didn't": "did not",
        "isn't": "is not", "aren't": "are not", "wasn't": "was not", "weren't": "were not",
        "shouldn't": "should not", "couldn't": "could not", "wouldn't": "would not",
        "haven't": "have not", "hasn't": "has not", "hadn't": "had not"
    }
    for contraction, replacement in contractions.items():
        text = re.sub(r'\b' + re.escape(contraction) + r'\b', replacement, text)

    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    words = text.split()
    words = [word for word in words if word not in stop_words]

    tagged_words = pos_tag(words)
    words = [stemmer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in tagged_words]

    return ' '.join(words)


def predict_disease(input_text: str, svm_model, vectorizer, stop_words, stemmer, label_encoder):
    """Predict disease using SVM model and return top 3 predictions."""
    if not input_text.strip():
        return None

    cleaned_text = clean_text(input_text, stop_words, stemmer)
    if not cleaned_text:
        return None

    try:
        vectorized = vectorizer.transform([cleaned_text]).toarray()
        prediction = svm_model.predict(vectorized)[0]
        decision_scores = svm_model.decision_function(vectorized)[0]
        
        # Get top 3 predictions
        top_3_indices = np.argsort(decision_scores)[::-1][:3]
        classes = svm_model.classes_
        
        top_predictions = []
        for idx in top_3_indices:
            score = decision_scores[idx]
            # Convert decision function to 0-1 probability using sigmoid
            confidence = 1 / (1 + np.exp(-score))
            top_predictions.append({
                "disease": classes[idx],
                "confidence": float(confidence)
            })
        
        return top_predictions
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None


def main() -> None:
    st.set_page_config(page_title="Symptom Checker", page_icon="🩺", layout="wide")
    st.title("🩺 Disease Symptom Checker")
    st.markdown(
        "Use the symptom matcher to compare your description with disease examples, or browse the known disease list from the dataset."
    )

    csv_path = Path(__file__).resolve().parent / "Symptom2Disease.csv"
    svm_model_path = Path(__file__).resolve().parent / "svm_model_tfidf.pkl"
    vectorizer_path = Path(__file__).resolve().parent / "tfidf_vectorizer.pkl"
    label_encoder_path = Path(__file__).resolve().parent / "label_encoder_normalize_reproducabled.pkl"

    uploaded_file = None
    df = None

    if not csv_path.exists():
        st.error(
            "The dataset file `Symptom2Disease.csv` is missing from the app folder."
        )
        uploaded_file = st.file_uploader(
            "Upload Symptom2Disease.csv", type=["csv"], help="Upload the same dataset file used locally."
        )
        if uploaded_file is not None:
            df = load_dataset_from_bytes(uploaded_file.getvalue())
        else:
            st.info(
                "To use the app, upload the CSV file or add `Symptom2Disease.csv` to the same folder as Streamlit.py."
            )
            st.stop()
    else:
        df = load_dataset_from_path(str(csv_path))

    svm_model, vectorizer = load_svm_model_and_vectorizer(str(svm_model_path), str(vectorizer_path))
    
    if not svm_model or not vectorizer:
        st.error("Failed to load the SVM model or vectorizer. Make sure they are saved in the same folder.")
        st.stop()

    try:
        label_encoder = joblib.load(str(label_encoder_path))
    except FileNotFoundError:
        st.error(f"Label encoder not found at {label_encoder_path}")
        st.stop()

    stop_words, stemmer = get_text_cleaner()
    if stop_words is None or stemmer is None:
        st.stop()

    diseases = sorted(df["disease"].unique())
    total_samples = len(df)

    st.sidebar.markdown("### Dataset overview")
    st.sidebar.metric("Known diseases", len(diseases))
    st.sidebar.metric("Symptom examples", total_samples)

    tab1, tab2 = st.tabs(["🔍 Check Symptoms", "📋 Known Diseases"])

    if tab1:
        st.subheader("Check your symptoms")
        st.markdown("Describe how you feel and the app will compare your text against known symptom examples.")

        left, right = st.columns([2, 1])
        with left:
            user_input = st.text_area(
                "Your symptoms",
                height=220,
                placeholder="e.g. fever, cough, headache, nausea, muscle pain...",
            )
            submit = st.button("Predict likely disease")

        with right:
            st.info("Tip: use words that describe your main symptoms clearly.")
            st.write("### Quick Stats")
            st.write(f"- Known diseases: **{len(diseases)}**")
            st.write(f"- Total dataset examples: **{total_samples}**")

        if submit:
            if not user_input.strip():
                st.warning("Please describe your symptoms before checking.")
            else:
                with st.spinner("Analyzing symptoms with ML model..."):
                    result = predict_disease(user_input, svm_model, vectorizer, stop_words, stemmer, label_encoder)

                if result is None:
                    st.error(
                        "Could not process your input. Try using clearer symptom descriptions."
                    )
                else:
                    st.success("🎯 Top 3 Disease Predictions")
                    
                    for i, pred in enumerate(result, 1):
                        disease = pred["disease"]
                        confidence = pred["confidence"]
                        
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.write(f"**{i}. {disease}**")
                            st.progress(confidence)
                        with col2:
                            st.metric("", f"{confidence * 100:.1f}%")

                    st.markdown("---")
                    best_disease = result[0]["disease"]
                    st.write(f"### Sample symptoms for **{best_disease}**")
                    examples = df[df["disease"] == best_disease]["symptoms"].head(5).tolist()
                    if examples:
                        for example in examples:
                            st.write(f"- {example}")
                    else:
                        st.info(f"No examples found for {best_disease} in the dataset.")

        st.markdown("---")
        st.caption(
            "This app uses a trained SVM model with TF-IDF vectorization to classify diseases from symptom descriptions. It is a demo only and not medical advice."
        )

    with tab2:
        st.subheader("Known Diseases")
        st.markdown("Browse the disease classes the model knows from the dataset.")

        search = st.text_input("Search diseases", placeholder="Type a name or keyword...")
        filtered = [d for d in diseases if search.lower() in d.lower()] if search else diseases

        st.write(f"Found **{len(filtered)}** disease names.")
        st.write(
            "Use the checkbox below to expand sample symptom descriptions for each disease."
        )

        if filtered:
            cols = st.columns([3, 1])
            with cols[0]:
                st.dataframe(pd.DataFrame({"Disease": filtered}), use_container_width=True)

            with cols[1]:
                st.write("### Quick view")
                if len(filtered) <= 10:
                    for disease in filtered[:10]:
                        st.write(f"- {disease}")
                else:
                    st.write("Showing the first 10 diseases.")
        else:
            st.info("No diseases matched your search.")

        with st.expander("Show sample symptom descriptions for diseases"):
            for disease in filtered[:20]:
                sample_texts = df[df["disease"] == disease]["symptoms"].head(3).tolist()
                if sample_texts:
                    st.write(f"**{disease}**")
                    for sample in sample_texts:
                        st.write(f"- {sample}")

        st.markdown("---")
        st.caption(
            "The known disease list is derived from the unique disease labels in Symptom2Disease.csv."
        )


if __name__ == "__main__":
    main()

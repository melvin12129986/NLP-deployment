import streamlit as st
import joblib
import os

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Health Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# CUSTOM CSS (Modern Glassmorphism & Dark Theme Styling)
# -----------------------------------------------------------------------------
st.markdown("""
    <style>
    /* Main Background Accent */
    .stApp {
        background-color: #0f172a;
        color: #f8fafc;
    }
    
    /* Header Container */
    .main-header {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        padding: 2rem;
        border-radius: 1rem;
        border: 1px solid #334155;
        margin-bottom: 2rem;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);
    }
    
    /* Custom Result Card */
    .result-card {
        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 100%);
        padding: 1.5rem;
        border-radius: 0.75rem;
        border: 1px solid #6366f1;
        text-align: center;
        margin-top: 1rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    
    .result-title {
        color: #a5b4fc;
        font-size: 0.875rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .result-value {
        color: #ffffff;
        font-size: 2rem;
        font-weight: 700;
        margin-top: 0.5rem;
    }

    /* Info Badge */
    .info-badge {
        background-color: #1e293b;
        border-left: 4px solid #38bdf8;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# MODEL LOADING (Cached for Speed)
# -----------------------------------------------------------------------------
@st.cache_resource
def load_models():
    try:
        model = joblib.load("svm_model_tfidf.pkl")
        vectorizer = joblib.load("tfidf_vectorizer.pkl")
        encoder = joblib.load("label_encoder_normalize_reproducabled.pkl")
        return model, vectorizer, encoder
    except Exception as e:
        return None, None, None

model, vectorizer, encoder = load_models()

# -----------------------------------------------------------------------------
# SIDEBAR NAVIGATION & DETAILS
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("🩺 AI Health Hub")
    st.markdown("---")
    
    st.subheader("About Model")
    st.write("**Architecture:** Support Vector Machine (SVM)")
    st.write("**Feature Extraction:** TF-IDF Vectorizer")
    st.write("**Task:** Multi-Class Disease Prediction")
    
    st.markdown("---")
    st.warning("⚠️ **Disclaimer:** This tool is for informational and educational purposes only. Always consult a certified healthcare professional for medical advice.")

# -----------------------------------------------------------------------------
# MAIN LAYOUT
# -----------------------------------------------------------------------------
# Header Banner
st.markdown("""
    <div class="main-header">
        <h1 style="margin:0; font-size: 2.2rem; font-weight: 800; color: #ffffff;">
            Symptom-to-Disease Diagnostic Assistant
        </h1>
        <p style="margin-top: 0.5rem; color: #94a3b8; font-size: 1rem;">
            Enter your current physical symptoms below to get an instant preliminary ML prediction.
        </p>
    </div>
""", unsafe_allow_html=True)

if model is None:
    st.error("Error: Could not load model files (`.pkl`). Please verify file paths in your repository.")
    st.stop()

# Two-Column Layout (Left: Inputs, Right: Predictions)
col1, col2 = st.columns([3, 2], gap="large")

with col1:
    st.subheader("Describe Your Symptoms")
    
    user_input = st.text_area(
        label="Symptom Description",
        placeholder="e.g., High fever, severe headache, muscle pain, skin rash, joint pain...",
        height=180,
        help="Type multiple symptoms separated by commas or in natural sentences for accurate results."
    )
    
    analyze_btn = st.button("🔍 Analyze Symptoms", type="primary", use_container_width=True)

with col2:
    st.subheader("Diagnostic Results")
    
    if analyze_btn:
        if not user_input.strip():
            st.warning("Please enter at least one symptom to analyze.")
        else:
            with st.spinner("Analyzing text & running inference..."):
                # Inference steps
                input_vec = vectorizer.transform([user_input])
                pred_encoded = model.predict(input_vec)
                predicted_disease = encoder.inverse_transform(pred_encoded)[0]
                
            # Display Modern Prediction Card
            st.markdown(f"""
                <div class="result-card">
                    <div class="result-title">Predicted Condition</div>
                    <div class="result-value">{predicted_disease}</div>
                </div>
            """, unsafe_allow_html=True)
            
            # Additional Context Container
            st.markdown(f"""
                <div class="info-badge">
                    <strong style="color: #38bdf8;">Next Steps:</strong><br/>
                    Review symptoms corresponding to <strong>{predicted_disease}</strong> with a medical practitioner.
                </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Input your symptoms in the field on the left and click **Analyze Symptoms** to generate a diagnostic prediction.")
import re
from collections import Counter
from pathlib import Path

import pandas as pd
import streamlit as st


@st.cache_data
def load_dataset() -> pd.DataFrame:
    csv_path = Path(__file__).resolve().parent / "Symptom2Disease.csv"
    df = pd.read_csv(csv_path, header=None, names=["id", "disease", "symptoms"])
    df["disease"] = df["disease"].astype(str).str.strip()
    df["symptoms"] = df["symptoms"].astype(str).str.strip()
    return df


def normalize_text(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = [token for token in text.split() if len(token) > 1]
    return tokens


def score_disease(input_text: str, df: pd.DataFrame) -> list[tuple[str, int]]:
    input_tokens = normalize_text(input_text)
    if not input_tokens:
        return []

    input_counter = Counter(input_tokens)
    disease_scores: dict[str, int] = {}

    for disease, group in df.groupby("disease"):
        score = 0
        for sample in group["symptoms"]:
            sample_tokens = normalize_text(sample)
            overlap = sum(min(input_counter[word], sample_tokens.count(word)) for word in set(sample_tokens))
            score += overlap
        disease_scores[disease] = score

    ranked = sorted(disease_scores.items(), key=lambda pair: pair[1], reverse=True)
    return [(disease, score) for disease, score in ranked if score > 0]


def main() -> None:
    st.set_page_config(page_title="Symptom Checker", page_icon="🩺", layout="wide")
    st.title("🩺 Disease Symptom Checker")
    st.markdown(
        "Use the symptom matcher to compare your description with disease examples, or browse the known disease list from the dataset."
    )

    df = load_dataset()
    diseases = sorted(df["disease"].unique())
    total_samples = len(df)
    disease_counts = Counter(df["disease"]).most_common(3)

    st.sidebar.title("Navigation")
    st.sidebar.write("Choose a page and explore the dataset.")
    page = st.sidebar.radio("Go to", ["Check Symptoms", "Known Diseases"])
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Dataset overview")
    st.sidebar.metric("Known diseases", len(diseases))
    st.sidebar.metric("Symptom examples", total_samples)

    if page == "Check Symptoms":
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
            st.write("### Fast facts")
            st.write(f"- Known diseases: **{len(diseases)}**")
            st.write(f"- Total dataset examples: **{total_samples}**")
            st.write("### Most frequent diseases")
            for disease, count in disease_counts:
                st.write(f"- **{disease}**: {count} examples")

        if submit:
            if not user_input.strip():
                st.warning("Please describe your symptoms before checking.")
            else:
                with st.spinner("Matching your symptoms..."):
                    predictions = score_disease(user_input, df)

                if not predictions:
                    st.error(
                        "No strong match was found. Try using more detailed or different symptoms."
                    )
                else:
                    top_matches = predictions[:5]
                    st.success("Possible disease matches")
                    for disease, score in top_matches:
                        st.markdown(f"**{disease}** — score: {score}")

                    st.markdown("---")
                    best_disease = top_matches[0][0]
                    st.write(f"### Sample symptom descriptions for **{best_disease}**")
                    examples = df[df["disease"] == best_disease]["symptoms"].head(5).tolist()
                    for example in examples:
                        st.write(f"- {example}")

        st.markdown("---")
        st.caption(
            "This app uses a simple text-overlap matcher on symptom descriptions from the dataset. It is a demo only and not medical advice."
        )

    else:
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

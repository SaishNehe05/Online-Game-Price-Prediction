"""Streamlit interface for estimating a Steam game's observed price."""
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "best_model.joblib"
METADATA_PATH = ROOT / "models" / "model_metadata.json"

st.set_page_config(page_title="Steam Game Price Predictor", page_icon="🎮", layout="wide")
st.title("🎮 Steam Game Price Predictor")
st.caption("Estimate a game's observed price from selected game characteristics. This is an educational ML prototype, not a live Steam price checker.")

if not MODEL_PATH.exists() or not METADATA_PATH.exists():
    st.warning("The trained model has not been generated yet.")
    st.markdown("Run these commands from the project root:")
    st.code("python scripts/download_data.py\npython src/train_model.py\nstreamlit run app.py", language="bash")
    st.stop()

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return model, metadata

model, metadata = load_artifacts()
left, right = st.columns([1.15, 0.85], gap="large")
with left:
    st.subheader("Enter game details")
    genre_options = metadata.get("genre_options", ["Unknown"])
    review_options = metadata.get("review_options", ["Unknown"])
    default_genre = genre_options.index("Unknown") if "Unknown" in genre_options else 0
    default_review = review_options.index("Unknown") if "Unknown" in review_options else 0
    with st.form("prediction_form"):
        genre = st.selectbox("Genre(s)", genre_options, index=default_genre, help="Choose a genre value that appears in the training dataset.")
        launch_year = st.number_input(
            "Launch year", min_value=int(metadata.get("launch_year_min", 1970)),
            max_value=int(metadata.get("launch_year_max", 2100)),
            value=int(min(max(2020, int(metadata.get("launch_year_min", 1970))), int(metadata.get("launch_year_max", 2100)))),
            step=1,
        )
        reviews = st.number_input("Number of reviews", min_value=0, max_value=int(max(100, metadata.get("reviews_max", 1000000))), value=100, step=10)
        review_summary = st.selectbox("General review summary", review_options, index=default_review)
        submitted = st.form_submit_button("Estimate game price", type="primary", use_container_width=True)

    if submitted:
        sample = pd.DataFrame([{
            "genres": genre,
            "launch_year": float(launch_year),
            "reviews": float(reviews),
            "general_reviews": review_summary,
        }])
        estimated_price = max(0.0, float(model.predict(sample)[0]))
        st.success("Estimated observed price")
        st.metric("Predicted price", f"€{estimated_price:,.2f}")
        st.caption("This is a model estimate based on a dataset collected in March 2025. It is not a current storefront quote.")

with right:
    st.subheader("Model information")
    st.write(f"**Selected model:** {metadata.get('selected_model', 'Not available')}")
    st.write(f"**Training records:** {metadata.get('train_rows', 'Not available'):,}" if isinstance(metadata.get('train_rows'), int) else "**Training records:** Not available")
    st.write(f"**Clean records:** {metadata.get('clean_rows', 'Not available'):,}" if isinstance(metadata.get('clean_rows'), int) else "**Clean records:** Not available")
    if metadata.get("test_mae_eur") is not None:
        st.metric("Held-out test MAE", f"€{metadata['test_mae_eur']:.2f}")
        st.metric("Held-out test RMSE", f"€{metadata['test_rmse_eur']:.2f}")
        st.metric("Held-out test R²", f"{metadata['test_r2']:.3f}")
    st.markdown("**Model inputs**")
    st.write("Genre, launch year, review count, and general review summary.")
    st.markdown("**Excluded to reduce leakage**")
    st.write("The current price is the target. `discount` and `original_price` are not input features.")
    st.markdown("**Dataset**")
    st.markdown("[Steam Games - UOC Project on Zenodo](https://zenodo.org/records/15121027)")

with st.expander("How should I interpret the result?"):
    st.write("The model learns associations present in the training dataset. The estimate may be inaccurate for games unlike those in the dataset, and Steam prices change over time and by country. Use the official Steam store to check a real game's current price.")

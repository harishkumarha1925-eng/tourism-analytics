# app.py (replace your current file)
import streamlit as st
import pandas as pd
import joblib
import os
from src.data_loader import load_raw, build_consolidated
from src.recommenders import simple_svd_recommender

st.set_page_config(layout="wide", page_title="Tourism Analytics")
st.title("Tourism Experience Analytics")

@st.cache_data
def get_data():
    """
    Load data via our loader. If loader returned empty or missing,
    try to fall back to a small committed CSV 'data/cleaned_small.csv'
    so Streamlit Cloud has something to show.
    """
    try:
        dfs = load_raw("data")
        df = build_consolidated(dfs)
    except Exception as e:
        st.error("Data loader raised an exception. See the logs.")
        st.write("Safe error message:", str(e))
        df = pd.DataFrame()

    # If df empty, fallback to a small sample CSV if present in repo
    if (df is None or df.empty) and os.path.exists("data/cleaned_small.csv"):
        try:
            df = pd.read_csv("data/cleaned_small.csv")
        except Exception as e:
            st.error("Failed to load fallback sample CSV. See logs.")
            st.write("Safe error:", str(e))
            df = pd.DataFrame()

    return df if df is not None else pd.DataFrame()

df = get_data()

# Show dataset sample button
st.sidebar.header("Actions")
if st.sidebar.button("Show sample data"):
    if df.empty:
        st.warning("No dataset available. Add a small sample CSV (data/cleaned_small.csv) to demo the app, or check logs.")
    else:
        st.dataframe(df.head(200))

st.sidebar.header("Recommendations")

# Defensive: ensure we only reference UserId if present
if df.empty or 'UserId' not in df.columns or df['UserId'].dropna().shape[0] == 0:
    st.sidebar.info("No user data available for recommendations. Add data/cleaned_small.csv or configure loader to have data files.")
else:
    # pick a safe default sample uid
    try:
        sample_uid = int(df['UserId'].dropna().sample(1).iloc[0])
    except Exception:
        sample_uid = int(df['UserId'].dropna().iloc[0])
    user_id = int(st.sidebar.number_input("UserId for recommendations", value=sample_uid, step=1))
    if st.sidebar.button("Get SVD recommendations"):
        recs = simple_svd_recommender(df, user_id, top_k=10)
        if not recs:
            st.sidebar.write("No recommendations found for this user.")
        else:
            st.sidebar.write("Recommended AttractionIds:", recs)
            st.write(df[df["AttractionId"].isin(recs)][["AttractionId","Attraction","AttractionType","AttractionCityName"]].drop_duplicates())

# Prediction area (safe checks)
st.header("Predict rating & visit mode (simple)")
col1, col2, col3 = st.columns(3)
year = col1.number_input("VisitYear", value=2023, step=1)
month = col2.number_input("VisitMonth", value=7, min_value=1, max_value=12)
user_visits = col3.number_input("User total visits", value=1, min_value=0)
attr_visits = st.number_input("Attraction total visits", value=1, min_value=0)

# try to load models, but don't crash if they're missing
try:
    reg = joblib.load("models/regressor_joblib.pkl")
except Exception:
    reg = None
try:
    clf = joblib.load("models/classifier_joblib.pkl")
    le = joblib.load("models/label_encoder.pkl")
except Exception:
    clf = None
    le = None

if st.button("Predict rating"):
    if reg is None:
        st.error("No regressor found. Run `python train.py` locally to create models and push models/ (or host them elsewhere).")
    else:
        import pandas as pd
        X = pd.DataFrame([[year, month, user_visits, attr_visits, 0, 0]],
                         columns=["VisitYear","VisitMonth","user_total_visits","attraction_total_visits","user_avg_rating","attraction_avg_rating"])
        pred = reg.predict(X)[0]
        st.success(f"Predicted rating: {pred:.2f}")

if st.button("Predict visit mode"):
    if clf is None or le is None:
        st.error("No classifier/label encoder found. Run `python train.py` locally to create models.")
    else:
        import pandas as pd
        Xc = pd.DataFrame([[year, month, user_visits, attr_visits, 0, 0]],
                          columns=["VisitYear","VisitMonth","user_total_visits","attraction_total_visits","user_avg_rating","attraction_avg_rating"])
        pred_c = clf.predict(Xc)[0]
        st.success(f"Predicted Visit Mode: {le.inverse_transform([pred_c])[0]}")

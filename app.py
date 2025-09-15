# app.py (robust, auto-download fallback)
import streamlit as st
import pandas as pd
import joblib
import os
import io
import requests
from src.data_loader import load_raw, build_consolidated
from src.recommenders import simple_svd_recommender

st.set_page_config(layout="wide", page_title="Tourism Analytics")
st.title("Tourism Experience Analytics")

@st.cache_data
def get_data():
    # 1) try loader (local data/ files)
    try:
        dfs = load_raw("data")
        df = build_consolidated(dfs)
        if df is not None and not df.empty:
            return df
    except Exception as e:
        # loader can fail in Cloud if files missing; we'll fallback
        st.write("Loader error (safe):", str(e))

    # 2) try local fallback sample if present
    local_sample = "data/cleaned_small.csv"
    if os.path.exists(local_sample):
        try:
            df = pd.read_csv(local_sample)
            if not df.empty:
                return df
        except Exception as e:
            st.write("Failed to read local sample (safe):", str(e))

    # 3) try to download sample from GitHub raw URL (your repo)
    gh_raw = "https://raw.githubusercontent.com/harishkumarha1925-eng/tourism-analytics/main/data/cleaned_small.csv"
    try:
        r = requests.get(gh_raw, timeout=10)
        if r.status_code == 200 and r.content:
            df = pd.read_csv(io.StringIO(r.text))
            if not df.empty:
                return df
    except Exception as e:
        st.write("Could not download sample from GitHub (safe):", str(e))

    # 4) final fallback: tiny builtin sample so UI works
    sample = pd.DataFrame([
        {"UserId":1,"AttractionId":1001,"Rating":5,"VisitYear":2023,"VisitMonth":7,"VisitMode":"Family","Attraction":"Beach A","AttractionType":"Beaches","AttractionCityName":"CityX"},
        {"UserId":2,"AttractionId":1002,"Rating":4,"VisitYear":2023,"VisitMonth":7,"VisitMode":"Couples","Attraction":"Museum B","AttractionType":"Museum","AttractionCityName":"CityY"},
        {"UserId":3,"AttractionId":1003,"Rating":3,"VisitYear":2022,"VisitMonth":10,"VisitMode":"Friends","Attraction":"Park C","AttractionType":"Park","AttractionCityName":"CityZ"},
        {"UserId":4,"AttractionId":1004,"Rating":5,"VisitYear":2022,"VisitMonth":11,"VisitMode":"Solo","Attraction":"Temple D","AttractionType":"Historical","AttractionCityName":"CityX"},
        {"UserId":5,"AttractionId":1005,"Rating":4,"VisitYear":2023,"VisitMonth":1,"VisitMode":"Business","Attraction":"Attraction E","AttractionType":"Monument","AttractionCityName":"CityY"},
    ])
    return sample

# load data
df = get_data()

# Data preview button
st.sidebar.header("Actions")
if st.sidebar.button("Show sample data"):
    if df is None or df.empty:
        st.warning("No dataset available. Add data/cleaned_small.csv to repo or check logs.")
    else:
        st.dataframe(df.head(200))

# Recommendations UI
st.sidebar.header("Recommendations")
if df is None or df.empty or 'UserId' not in df.columns or df['UserId'].dropna().shape[0] == 0:
    st.sidebar.info("No user data available for recommendations. Add data/cleaned_small.csv or ensure loader files are present.")
else:
    # choose user id safely
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

# Prediction UI (models optional)
st.header("Predict rating & visit mode (simple)")
col1, col2, col3 = st.columns(3)
year = col1.number_input("VisitYear", value=2023, step=1)
month = col2.number_input("VisitMonth", value=7, min_value=1, max_value=12)
user_visits = col3.number_input("User total visits", value=1, min_value=0)
attr_visits = st.number_input("Attraction total visits", value=1, min_value=0)

# load models safely if present
reg = None
clf = None
le = None
try:
    if os.path.exists("models/regressor_joblib.pkl"):
        reg = joblib.load("models/regressor_joblib.pkl")
    if os.path.exists("models/classifier_joblib.pkl"):
        clf = joblib.load("models/classifier_joblib.pkl")
    if os.path.exists("models/label_encoder.pkl"):
        le = joblib.load("models/label_encoder.pkl")
except Exception as e:
    st.write("Model loading warning (safe):", str(e))

if st.button("Predict rating"):
    import pandas as _pd
    X = _pd.DataFrame([[year, month, user_visits, attr_visits, 0, 0]],
                      columns=["VisitYear","VisitMonth","user_total_visits","attraction_total_visits","user_avg_rating","attraction_avg_rating"])
    if reg is None:
        st.error("No regressor model found. Run `python train.py` locally and add models/ to repo or host models remotely.")
    else:
        pred = reg.predict(X)[0]
        st.success(f"Predicted rating: {pred:.2f}")

if st.button("Predict visit mode"):
    import pandas as _pd
    Xc = _pd.DataFrame([[year, month, user_visits, attr_visits, 0, 0]],
                       columns=["VisitYear","VisitMonth","user_total_visits","attraction_total_visits","user_avg_rating","attraction_avg_rating"])
    if clf is None or le is None:
        st.error("No classifier/label encoder found. Run `python train.py` locally and add models/ to repo or host models remotely.")
    else:
        pred_c = clf.predict(Xc)[0]
        st.success(f"Predicted Visit Mode: {le.inverse_transform([pred_c])[0]}")


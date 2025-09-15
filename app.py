# app.py
import streamlit as st
import pandas as pd
import joblib
from src.data_loader import load_raw, build_consolidated
from src.recommenders import simple_svd_recommender, user_item_matrix

st.set_page_config(layout="wide", page_title="Tourism Analytics")

st.title("Tourism Experience Analytics")

@st.cache_data
def get_data():
    dfs = load_raw("data")
    df = build_consolidated(dfs)
    return df

df = get_data()

st.sidebar.header("Actions")
if st.sidebar.button("Show sample data"):
    st.dataframe(df.head(200))

st.sidebar.header("Recommendations")
user_id = int(st.sidebar.number_input("UserId for recommendations", value=int(df['UserId'].dropna().sample(1).iloc[0])))
if st.sidebar.button("Get SVD recommendations"):
    recs = simple_svd_recommender(df, user_id, top_k=10)
    st.sidebar.write("Recommended AttractionIds:", recs)
    if recs:
        st.write(df[df["AttractionId"].isin(recs)][["AttractionId","Attraction","AttractionType","AttractionCityName"]].drop_duplicates())

st.sidebar.header("Predictions")
# Load models
try:
    reg = joblib.load("models/regressor_joblib.pkl")
    clf = joblib.load("models/classifier_joblib.pkl")
    le = joblib.load("models/label_encoder.pkl")
    st.sidebar.success("Models loaded")
except Exception:
    reg = clf = le = None
    st.sidebar.info("No trained models found. Run `python train.py`")

st.header("Predict rating & visit mode (simple)")
col1, col2, col3 = st.columns(3)
year = col1.number_input("VisitYear", value=2023, step=1)
month = col2.number_input("VisitMonth", value=7, min_value=1, max_value=12)
user_visits = col3.number_input("User total visits", value=1, min_value=0)
attr_visits = st.number_input("Attraction total visits", value=1, min_value=0)

if st.button("Predict rating"):
    if reg is not None:
        X = pd.DataFrame([[year, month, user_visits, attr_visits, 0, 0]], columns=["VisitYear","VisitMonth","user_total_visits","attraction_total_visits","user_avg_rating","attraction_avg_rating"])
        pred = reg.predict(X)[0]
        st.success(f"Predicted rating: {pred:.2f}")
    else:
        st.error("Train models first: python train.py")

if st.button("Predict visit mode"):
    if clf is not None and le is not None:
        Xc = pd.DataFrame([[year, month, user_visits, attr_visits, 0, 0]], columns=["VisitYear","VisitMonth","user_total_visits","attraction_total_visits","user_avg_rating","attraction_avg_rating"])
        pred_c = clf.predict(Xc)[0]
        st.success(f"Predicted Visit Mode: {le.inverse_transform([pred_c])[0]}")
    else:
        st.error("Train models first: python train.py")

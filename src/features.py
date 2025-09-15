# src/features.py
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np

def add_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    # user aggregates
    user_agg = df.groupby("UserId").agg(
        user_total_visits=("TransactionId", "count"),
        user_avg_rating=("Rating", "mean")
    ).reset_index()
    df = df.merge(user_agg, on="UserId", how="left")

    # attraction aggregates
    attr_agg = df.groupby(["AttractionId", "Attraction"]).agg(
        attraction_total_visits=("TransactionId", "count"),
        attraction_avg_rating=("Rating", "mean")
    ).reset_index()
    df = df.merge(attr_agg, on=["AttractionId", "Attraction"], how="left")
    return df

def create_basic_feature_matrix(df: pd.DataFrame, categorical_cols=None, numeric_cols=None):
    """
    Returns X_df (pandas) and fitted OneHotEncoder.
    This code is compatible with different sklearn versions (sparse vs sparse_output).
    """
    df = df.copy()
    categorical_cols = categorical_cols or ["AttractionType", "UserContinent", "UserCountry", "VisitModeName"]
    numeric_cols = numeric_cols or ["VisitYear", "VisitMonth", "user_total_visits", "attraction_total_visits", "user_avg_rating", "attraction_avg_rating"]

    # numeric part
    X_num = pd.DataFrame()
    for c in numeric_cols:
        if c in df.columns:
            X_num[c] = pd.to_numeric(df[c].fillna(0), errors="coerce").astype(float)
        else:
            X_num[c] = 0.0

    # categorical part
    # defensive: keep only columns that exist
    categorical_cols = [c for c in (categorical_cols or []) if c in df.columns]
    if categorical_cols:
        cat_df = df[categorical_cols].fillna("NA").astype(str)
        # Create OneHotEncoder in a sklearn-version resilient way
        try:
            # sklearn >= 1.2
            enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            # older sklearn
            enc = OneHotEncoder(handle_unknown="ignore", sparse=False)

        enc_arr = enc.fit_transform(cat_df)
        enc_cols = enc.get_feature_names_out(categorical_cols)
        X_cat = pd.DataFrame(enc_arr, columns=enc_cols, index=df.index)
        X = pd.concat([X_num, X_cat], axis=1)
    else:
        enc = None
        X = X_num

    return X, enc

def label_encode_visitmode(df):
    from sklearn.preprocessing import LabelEncoder
    df = df.copy()
    le = LabelEncoder()
    df = df.dropna(subset=["VisitModeName"])
    df["visit_mode_label"] = le.fit_transform(df["VisitModeName"].astype(str))
    return df, le


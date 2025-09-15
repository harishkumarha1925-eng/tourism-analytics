# src/cleaning.py
import pandas as pd

def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # strip column names and standardize
    df.columns = [c.strip() for c in df.columns]
    # replace common placeholder values with NaN
    df = df.replace({'-': pd.NA, '': pd.NA})
    # try infer_objects with copy=False if available, else fallback
    try:
        # pandas >= some versions accept copy arg; used to avoid the FutureWarning
        df = df.infer_objects(copy=False)
    except TypeError:
        df = df.infer_objects()

    # date sanity: VisitMonth/VisitYear -> numeric
    if "VisitMonth" in df.columns:
        df["VisitMonth"] = pd.to_numeric(df["VisitMonth"], errors="coerce").astype("Int64")
    if "VisitYear" in df.columns:
        df["VisitYear"] = pd.to_numeric(df["VisitYear"], errors="coerce").astype("Int64")
    # enforce integer ids where sensible
    for col in ["UserId", "AttractionId", "TransactionId"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    return df

def drop_low_information_cols(df, threshold=0.98):
    to_drop = []
    for col in df.columns:
        try:
            top_freq = df[col].value_counts(normalize=True, dropna=False).values[0]
            if top_freq >= threshold:
                to_drop.append(col)
        except Exception:
            continue
    return df.drop(columns=to_drop, errors="ignore")



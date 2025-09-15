# src/modeling.py
import joblib
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
import math
import numpy as np

def _safe_rmse(y_true, y_pred):
    # handle multioutput shapes gracefully
    try:
        mse = mean_squared_error(y_true, y_pred)
        return math.sqrt(mse)
    except Exception:
        # fallback: compute elementwise then aggregate
        arr_true = np.array(y_true).ravel()
        arr_pred = np.array(y_pred).ravel()
        return float(np.sqrt(np.mean((arr_true - arr_pred) ** 2)))

def train_regression(X, y, save_path=None):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse = _safe_rmse(y_test, preds)
    if save_path:
        joblib.dump(model, save_path)
    return model, rmse

def train_classification(X, y, save_path=None):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds)
    if save_path:
        joblib.dump(clf, save_path)
    return clf, acc, report


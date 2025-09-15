# src/eval.py
import numpy as np
from sklearn.metrics import mean_squared_error

def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred, squared=False)

def precision_at_k(recommended, actual, k=10):
    recommended_k = recommended[:k]
    hits = sum([1 for r in recommended_k if r in actual])
    return hits / k

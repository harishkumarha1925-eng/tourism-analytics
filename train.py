# train.py
from src.data_loader import load_raw, build_consolidated, merge_updated_item
from src.cleaning import basic_clean
from src.features import add_aggregates, create_basic_feature_matrix, label_encode_visitmode
from src.modeling import train_regression, train_classification
import joblib
from pathlib import Path
import os

def main():
    data_dir = "data"
    Path("models").mkdir(exist_ok=True)
    # Ensure merged items
    merge_updated_item(data_dir)
    dfs = load_raw(data_dir)
    df = build_consolidated(dfs)
    df = basic_clean(df)
    df = add_aggregates(df)

    # Save cleaned dataset
    df.to_csv(Path(data_dir)/"cleaned_tourism_with_updated_items.csv", index=False)
    print("Saved cleaned dataset to data/cleaned_tourism_with_updated_items.csv")

    # Features for regression (simple starter)
    numeric_cols = ["VisitYear", "VisitMonth", "user_total_visits", "attraction_total_visits", "user_avg_rating", "attraction_avg_rating"]
    categorical_cols = ["AttractionType", "UserContinent", "UserCountry", "VisitModeName"]
    X, enc = create_basic_feature_matrix(df, categorical_cols=categorical_cols, numeric_cols=numeric_cols)
    y = df["Rating"]

    # train regression
    reg_model, reg_rmse = train_regression(X, y, save_path="models/regressor_joblib.pkl")
    print("Regression RMSE:", reg_rmse)

    # classification (VisitMode)
    df_mode, le = label_encode_visitmode(df)
    Xc, enc2 = create_basic_feature_matrix(df_mode, categorical_cols=categorical_cols, numeric_cols=numeric_cols)
    yc = df_mode["visit_mode_label"]
    clf, acc, report = train_classification(Xc, yc, save_path="models/classifier_joblib.pkl")
    joblib.dump(le, "models/label_encoder.pkl")
    joblib.dump(enc, "models/onehot_enc.pkl")
    print("Classification accuracy:", acc)
    print(report)

if __name__ == "__main__":
    main()

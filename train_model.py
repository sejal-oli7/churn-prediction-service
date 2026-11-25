import argparse

import joblib
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

CATEGORICAL = [
    "gender", "seniorcitizen", "partner", "dependents", "phoneservice",
    "multiplelines", "internetservice", "onlinesecurity", "onlinebackup",
    "deviceprotection", "techsupport", "streamingtv", "streamingmovies",
    "contract", "paperlessbilling", "paymentmethod",
]
NUMERICAL = ["tenure", "monthlycharges", "totalcharges"]


def main(data_path, out_path):
    df = pd.read_csv(data_path)
    df.columns = df.columns.str.lower().str.replace(" ", "_")
    for c in df.columns:
        if df[c].dtype == "object" or str(df[c].dtype) == "str":
            df[c] = df[c].str.lower().str.replace(" ", "_")

    df["totalcharges"] = pd.to_numeric(df["totalcharges"], errors="coerce").fillna(0)
    df["churn"] = (df["churn"] == "yes").astype(int)

    df_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
    y_train, y_test = df_train["churn"].values, df_test["churn"].values

    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(df_train[CATEGORICAL + NUMERICAL].to_dict(orient="records"))
    X_test = dv.transform(df_test[CATEGORICAL + NUMERICAL].to_dict(orient="records"))

    model = LogisticRegression(solver="liblinear", C=1.0, max_iter=1000)
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    print("Features      :", X_train.shape[1])
    print("Test accuracy :", round(accuracy_score(y_test, proba >= 0.5), 3))
    print("Test ROC AUC  :", round(roc_auc_score(y_test, proba), 3))

    assert len(dv.get_feature_names_out()) == model.n_features_in_, "dv/model mismatch!"
    joblib.dump({"dv": dv, "model": model}, out_path)
    print("Saved ->", out_path)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", default="churn_model.pkl")
    a = p.parse_args()
    main(a.data, a.out)

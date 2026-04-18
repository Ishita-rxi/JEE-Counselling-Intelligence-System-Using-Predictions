import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier

df = pd.read_csv("josaa_all_years.csv")

df = df.rename(columns={
    "Institute": "institute",
    "Program": "branch",
    "Quota": "quota",
    "Seat Type": "category",
    "Gender": "gender",
    "Closing Rank": "closing_rank",
    "Year": "year"
})

df = df[["year","institute","branch","quota","category","gender","closing_rank"]]

for col in ["institute","branch","quota","category","gender"]:
    df[col] = df[col].astype(str).str.lower().str.strip()

df["branch"] = df["branch"].str.replace(r"\s*\(.*\)", "", regex=True)

df["closing_rank"] = df["closing_rank"].astype(str).str.replace(",", "")
df = df[~df["closing_rank"].str.contains("P", na=False)]
df["closing_rank"] = pd.to_numeric(df["closing_rank"], errors="coerce")
df = df.dropna()
df["closing_rank"] = df["closing_rank"].astype(int)

df = df.sort_values(["institute","branch","category","quota","gender","year"])

df["prev_closing_rank"] = df.groupby(
    ["institute","branch","category","quota","gender"]
)["closing_rank"].shift(1)

df["cutoff_trend"] = df["closing_rank"] - df["prev_closing_rank"]

df = df.dropna()

rows = []

for _, row in df.iterrows():
    closing = row["closing_rank"]

    ranks = np.concatenate([
        np.linspace(1, closing, 20),
        np.linspace(closing, closing*1.5, 20),
        np.linspace(closing*1.5, closing*2, 10)
    ])

    for r in ranks:
        noise = np.random.normal(0, closing*0.2)
        label = 1 if r <= closing + noise else 0

        rows.append({
            "rank": int(r),
            "institute": row["institute"],
            "branch": row["branch"],
            "quota": row["quota"],
            "category": row["category"],
            "gender": row["gender"],
            "year": row["year"],
            "prev_closing_rank": row["prev_closing_rank"],
            "cutoff_trend": row["cutoff_trend"],
            "label": label
        })

train_df = pd.DataFrame(rows)

encoders = {}

for col in ["institute","branch","quota","category","gender"]:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col])
    encoders[col] = le

train = train_df[train_df["year"] < 2025]
test  = train_df[train_df["year"] == 2025]

X_train = train.drop(columns=["label"])
y_train = train["label"]

X_test = test.drop(columns=["label"])
y_test = test["label"]

param_grid = {
    "n_estimators":[100,200],
    "max_depth":[4,6],
    "learning_rate":[0.05,0.1]
}

grid = GridSearchCV(
    XGBClassifier(eval_metric='logloss'),
    param_grid,
    scoring="average_precision",
    cv=3,
    n_jobs=-1
)

grid.fit(X_train, y_train)

model = grid.best_estimator_

print("PR-AUC:", average_precision_score(y_test, model.predict_proba(X_test)[:,1]))

joblib.dump(model, "model.pkl")
joblib.dump(encoders, "encoders.pkl")

print("Model saved ✅")
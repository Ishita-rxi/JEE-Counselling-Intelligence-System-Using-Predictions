import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="JEE College Predictor", layout="wide")

# ======================
# LOAD MODEL + ENCODERS
# ======================
model = joblib.load("model.pkl")
encoders = joblib.load("encoders.pkl")

st.title("🎓 JEE College Predictor")

# ======================
# USER INPUT
# ======================
rank = st.number_input("Enter your Rank", min_value=1, step=1)

category = st.selectbox("Category", encoders["category"].classes_)
quota = st.selectbox("Quota", encoders["quota"].classes_)
gender = st.selectbox("Gender", encoders["gender"].classes_)

# ======================
# LOAD & CLEAN DATA
# ======================
#@st.cache_data
def load_data():
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

    df = df[[
        "year", "institute", "branch", "quota",
        "category", "gender", "closing_rank"
    ]]

    # ----------------------
    # CLEAN TEXT
    # ----------------------
    for col in ["institute","branch","quota","category","gender"]:
        df[col] = df[col].astype(str).str.lower().str.strip()

    df["branch"] = df["branch"].str.replace(r"\s*\(.*\)", "", regex=True)

    # ----------------------
    # 🔥 CLEAN closing_rank
    # ----------------------
    df["closing_rank"] = (
        df["closing_rank"]
        .astype(str)
        .str.replace(",", "", regex=False)
    )

    df = df[~df["closing_rank"].str.contains("P", na=False)]

    df["closing_rank"] = pd.to_numeric(df["closing_rank"], errors="coerce")
    df = df.dropna(subset=["closing_rank"])

    # FORCE numeric (important)
    df["closing_rank"] = df["closing_rank"].astype(float)

    # ----------------------
    # SORT BEFORE GROUPBY
    # ----------------------
    df = df.sort_values(
        ["institute","branch","category","quota","gender","year"]
    )

    # ----------------------
    # FEATURE ENGINEERING
    # ----------------------
    df["prev_closing_rank"] = df.groupby(
        ["institute","branch","category","quota","gender"]
    )["closing_rank"].shift(1)

    # 🔥 CRITICAL FIX (this avoids your error)
    df["prev_closing_rank"] = pd.to_numeric(
        df["prev_closing_rank"], errors="coerce"
    )

    df = df.dropna(subset=["prev_closing_rank"])

    # NOW safe subtraction
    df["cutoff_trend"] = df["closing_rank"] - df["prev_closing_rank"]

    return df

df = load_data()

# ======================
# LATEST DATA
# ======================
latest = df[df["year"] == df["year"].max()]

# ======================
# PREDICTION
# ======================
if st.button("Predict Colleges"):

    results = []

    for _, row in latest.iterrows():

        try:
            sample = pd.DataFrame([{
                "rank": rank,
                "institute": encoders["institute"].transform([row["institute"]])[0],
                "branch": encoders["branch"].transform([row["branch"]])[0],
                "quota": encoders["quota"].transform([row["quota"]])[0],
                "category": encoders["category"].transform([category])[0],
                "gender": encoders["gender"].transform([gender])[0],
                "year": row["year"],
                "prev_closing_rank": float(row["prev_closing_rank"]),
                "cutoff_trend": float(row["cutoff_trend"])
            }])

            prob = model.predict_proba(sample)[0][1]

            if prob >= 0.8:
                tag = "SAFE"
            elif prob >= 0.5:
                tag = "TARGET"
            else:
                tag = "DREAM"

            results.append({
                "Institute": row["institute"],
                "Branch": row["branch"],
                "Probability": round(prob, 3),
                "Type": tag
            })

        except Exception as e:
            continue

    if len(results) == 0:
        st.error("No predictions generated. Check data consistency.")
    else:
        res_df = pd.DataFrame(results)

        # sort + clean
        res_df = res_df.sort_values(by="Probability", ascending=False)
        res_df = res_df.drop_duplicates(subset=["Institute","Branch"])

        st.subheader("🎯 Top Recommendations")
        st.dataframe(res_df.head(20), use_container_width=True)

        # CATEGORY SPLIT
        st.subheader("📊 Category-wise")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("### 🟢 SAFE")
            st.dataframe(res_df[res_df["Type"]=="SAFE"].head(5))

        with col2:
            st.markdown("### 🟡 TARGET")
            st.dataframe(res_df[res_df["Type"]=="TARGET"].head(5))

        with col3:
            st.markdown("### 🔴 DREAM")
            st.dataframe(res_df[res_df["Type"]=="DREAM"].head(5))
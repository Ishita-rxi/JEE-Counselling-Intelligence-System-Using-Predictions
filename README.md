# 🎓 JEE College Predictor (Machine Learning Project)

An end-to-end machine learning application that predicts the probability of admission into engineering colleges based on JEE rank using historical JoSAA cutoff data. The system provides personalized recommendations categorized as Safe, Target, and Dream.

---

## 🚀 Overview

Traditional college predictors rely on rigid cutoff ranks. This project improves upon that by modeling **probabilistic admission likelihood**, accounting for variations and trends across years.

The system uses historical data, feature engineering, and an XGBoost classifier to estimate admission chances for different institute–branch combinations.

---

## 🧠 Key Features

- **Probabilistic Prediction** instead of fixed cutoffs  
- **Temporal Feature Engineering** (previous year cutoffs, trends)  
- **Precision–Recall AUC evaluation** for imbalanced classification  
- **Recommendation Engine**:
  - 🟢 Safe (high probability)
  - 🟡 Target (moderate probability)
  - 🔴 Dream (low probability)
- **Interactive Web Interface** using Streamlit  

---

## 🏗️ Tech Stack

- Python  
- Pandas, NumPy  
- Scikit-learn  
- XGBoost  
- Streamlit  
- Joblib  

---

## 📂 Project Structure
- app.py # Streamlit web app
- train_model.py # Model training pipeline
- data.csv # JoSAA dataset
- model.pkl # Trained ML model
- encoders.pkl # Label encoders
- requirements.txt # Dependencies
- README.md

---

## ⚙️ Methodology

### 1. Data Preprocessing
- Cleaned dataset (removed invalid ranks like 'P')
- Standardized categorical features (institute, branch, category, etc.)
- Converted ranks into numeric format

### 2. Feature Engineering
- Previous year closing rank  
- Cutoff trend (difference across years)  
- Synthetic data generation to simulate decision boundaries  

### 3. Model Training
- Model: **XGBoost Classifier**
- Validation: **Time-based split** (train on past, test on future)
- Evaluation metric: **Precision–Recall AUC**

### 4. Prediction & Recommendation
- Input: rank, category, quota, gender  
- Output: probability of admission  
- Results ranked and classified into Safe / Target / Dream  

---

## 📊 Sample Output

| Institute | Branch | Probability | Category |
|----------|--------|------------|----------|
| IIT X | CSE | 0.89 | SAFE |
| NIT Y | ECE | 0.68 | TARGET |
| IIT Z | ME | 0.41 | DREAM |

---

## 🎯 Key Learnings

- Handling **imbalanced datasets** using PR-AUC  
- Importance of **temporal validation** in real-world data  
- Feature engineering significantly impacts model performance  
- Efficient inference using **vectorized predictions**  

---

## 🚀 Future Improvements

- Add percentile-based predictions  
- Include seat availability data  
- Improve recommendation ranking logic  
- Deploy publicly (Streamlit Cloud / Render)  
- Add visual analytics dashboard  

---

## 📌 Author

Your Ishita Singh

---

## ⭐ Acknowledgment

Based on publicly available JoSAA cutoff data.

---

## 📎 Note

This project is intended for educational purposes and provides probabilistic estimates, not guaranteed admission outcomes.

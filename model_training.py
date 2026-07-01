# ===============================
# Credit Risk Scorecard Training
# ===============================

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


# -------------------------------
# 1. Load dataset
# -------------------------------
df = pd.read_csv("credit_risk_scorecard_dataset.csv")
print("Dataset loaded")


# -------------------------------
# 2. Identify columns
# -------------------------------
ID_COL = "customer_id"
TARGET_COL = "default_flag"   # your dataset uses this
DROP_COLS = [ID_COL, TARGET_COL]

print("Columns:", df.columns.tolist())


# -------------------------------
# 3. Split features and target
# -------------------------------
X = df.drop(DROP_COLS, axis=1)
y = df[TARGET_COL]


# -------------------------------
# 4. Train-test split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)


# -------------------------------
# 5. Scaling
# -------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# -------------------------------
# 6. Train model
# -------------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)


# -------------------------------
# 7. Evaluation
# -------------------------------
pd_test = model.predict_proba(X_test_scaled)[:, 1]
auc = roc_auc_score(y_test, pd_test)
print("ROC AUC:", round(auc, 4))


# -------------------------------
# 8. Save artifacts (CRITICAL)
# -------------------------------
joblib.dump(model, "credit_risk_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(X.columns.tolist(), "model_features.pkl")

print("Artifacts saved successfully")
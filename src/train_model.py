import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from joblib import dump
from data_preprocessing import preprocess

# -----------------------------
# Load data
# -----------------------------
DATA_PATH = "../data/raw/train.csv"
df = pd.read_csv(DATA_PATH)

y = df["Survived"]

# 🚨 DROP PassengerId HERE
X_raw = df.drop(columns=["Survived", "PassengerId"])

# -----------------------------
# Preprocess
# -----------------------------
X_processed, feature_columns = preprocess(
    X_raw,
    is_train=True
)

# -----------------------------
# Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_processed,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------
# Models
# -----------------------------
models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=6000,
            C=0.7,
            class_weight="balanced",
            solver="lbfgs"
        ))
    ]),

    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.9,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=400,
        max_depth=7,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
}

best_model = None
best_auc = 0
best_threshold = 0.5

print("\n================ MODEL RESULTS ================\n")

# -----------------------------
# Train & evaluate
# -----------------------------
for name, model in models.items():
    print(f"Training {name}")

    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]

    thresholds = np.arange(0.25, 0.70, 0.01)
    f1_scores = []

    for t in thresholds:
        preds = (probs >= t).astype(int)
        f1_scores.append(f1_score(y_test, preds))

    best_t = thresholds[np.argmax(f1_scores)]
    preds = (probs >= best_t).astype(int)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)
    cm = confusion_matrix(y_test, preds)

    print(f"ROC-AUC   : {auc:.4f}")
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1-Score  : {f1:.4f}")
    print(f"Threshold : {best_t:.2f}")
    print("Confusion Matrix:\n", cm)
    print("-" * 60)

    if auc > best_auc:
        best_auc = auc
        best_model = model
        best_threshold = best_t

# -----------------------------
# Save artifacts
# -----------------------------
os.makedirs("../models", exist_ok=True)

dump(best_model, "../models/titanic_model.joblib")
dump(feature_columns, "../models/feature_columns.joblib")
dump(best_threshold, "../models/best_threshold.joblib")

print("\n✅ FINAL MODEL SAVED")
print(f"Best ROC-AUC  : {best_auc:.4f}")
print(f"Best Threshold: {best_threshold:.2f}")

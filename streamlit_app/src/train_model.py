import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score
)

from joblib import dump
from data_preprocessing import preprocess

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv("../data/raw/train.csv")
y = df["Survived"]

X_proc, feature_columns = preprocess(
    df.drop(columns=["Survived"]),
    is_train=True
)
X = X_proc.drop(columns=["PassengerId"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
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
            max_iter=3000,
            class_weight="balanced"
        ))
    ]),
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42
    ),
    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(
            probability=True,
            class_weight="balanced"
        ))
    ])
}

best_model = None
best_threshold = 0.5
best_auc = 0

print("\n================ MODEL RESULTS ================\n")

# -----------------------------
# Train & Evaluate
# -----------------------------
for name, model in models.items():
    print(f"Training {name}")

    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]

    # ----- Threshold tuning (PER MODEL) -----
    thresholds = np.arange(0.30, 0.71, 0.02)
    f1_scores = []

    for t in thresholds:
        preds = (probs >= t).astype(int)
        f1_scores.append(f1_score(y_test, preds))

    model_best_threshold = thresholds[np.argmax(f1_scores)]
    model_best_f1 = max(f1_scores)

    preds = (probs >= model_best_threshold).astype(int)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)
    cm = confusion_matrix(y_test, preds)

    print(f"ROC-AUC   : {auc:.4f}")
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1-Score  : {model_best_f1:.4f}")
    print(f"Threshold : {model_best_threshold:.2f}")
    print("Confusion Matrix:\n", cm)
    print("-" * 55)

    # ----- Select best model by ROC-AUC -----
    if auc > best_auc:
        best_auc = auc
        best_model = model
        best_threshold = model_best_threshold

# -----------------------------
# Save final model
# -----------------------------
os.makedirs("../models", exist_ok=True)

dump(best_model, "../models/titanic_model.joblib")
dump(feature_columns, "../models/feature_columns.joblib")
dump(best_threshold, "../models/best_threshold.joblib")

print("\n✅ FINAL MODEL SAVED")
print(f"Best ROC-AUC  : {best_auc:.4f}")
print(f"Best Threshold: {best_threshold:.2f}")

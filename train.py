"""Train + compare models with SUBJECT-WISE cross-validation, then save the best one with SHAP plot.
Run: python train.py
"""
import os
import joblib
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from features import FEATURES

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data"
PATH = "data/parkinsons.data"
os.makedirs("data", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

df = pd.read_csv(PATH if os.path.exists(PATH) else URL)
df.to_csv(PATH, index=False) if not os.path.exists(PATH) else None

X, y = df[FEATURES], df["status"]
groups = df["name"].str.split("_").str[2]  # e.g. phon_R01_S01_1 -> S01 (subject id)
print(f"{len(df)} recordings | {groups.nunique()} subjects | PD ratio {y.mean():.2f}")

models = {
    "LogReg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
    "RandomForest": RandomForestClassifier(n_estimators=400, class_weight="balanced_subsample", random_state=42),
    "XGBoost": XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8,
                             eval_metric="logloss", random_state=42),
}

cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)  # no subject appears in train AND test
rows = []
for name, model in models.items():
    prob = cross_val_predict(model, X, y, groups=groups, cv=cv, method="predict_proba")[:, 1]
    pred = (prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    rows.append({"model": name, "ROC-AUC": roc_auc_score(y, prob), "sensitivity": tp / (tp + fn),
                 "specificity": tn / (tn + fp), "F1": f1_score(y, pred)})

results = pd.DataFrame(rows).round(3).sort_values("ROC-AUC", ascending=False)
print("\nSubject-wise 5-fold CV results:\n", results.to_string(index=False))
results.to_csv("outputs/results.csv", index=False)

best_name = results.iloc[0]["model"]
best = models[best_name].fit(X, y)
print(f"\nBest model: {best_name}")

# --- Explainability (model-agnostic SHAP on the PD probability) ---
background = X.sample(min(100, len(X)), random_state=42)
explainer = shap.Explainer(lambda d: best.predict_proba(pd.DataFrame(d, columns=FEATURES))[:, 1], background)
sv = explainer(X.sample(80, random_state=1))
shap.summary_plot(sv, show=False)
plt.tight_layout()
plt.savefig("outputs/shap_summary.png", dpi=150)

joblib.dump({"model": best, "name": best_name, "background": background}, "outputs/model.joblib")
print("Saved outputs/model.joblib, results.csv, shap_summary.png")
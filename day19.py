import warnings
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report, roc_auc_score, roc_curve)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# =============================================
# Day 19 - Model Evaluation Deep Dive
# =============================================


# --- Why Evaluation Matters ---
# accuracy alone can be misleading on imbalanced data
# a model that always predicts "healthy" on a 99% healthy dataset gets 99% accuracy
# but misses every real positive — precision, recall, AUC tell the full story


# --- Overfitting and Underfitting ---
# overfitting:   train accuracy high, test accuracy low — model memorized training data
# underfitting:  both low — model too simple to learn the pattern
# just right:    train and test accuracy close, both high — model learned the real pattern

cancer   = load_breast_cancer()
X_cancer = pd.DataFrame(cancer.data, columns=cancer.feature_names)
y_cancer = pd.Series(cancer.target)

X_train, X_test, y_train, y_test = train_test_split(X_cancer, y_cancer, test_size=0.2, random_state=42)

for depth in [1, 3, 10, 20]:
    model     = DecisionTreeClassifier(max_depth=depth, random_state=42)
    model.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc  = accuracy_score(y_test,  model.predict(X_test))
    gap       = train_acc - test_acc
    print(f"depth={depth:2d} | train={train_acc:.2f} | test={test_acc:.2f} | gap={gap:.2f}")
# depth=1 underfits | depth=3 is the sweet spot | depth=10+ overfits (train=1.00)


# --- Confusion Matrix ---
# breaks predictions into 4 buckets: TP, TN, FP, FN
# more honest than accuracy — shows exactly where the model is wrong

model_cm = DecisionTreeClassifier(max_depth=3, random_state=42)
model_cm.fit(X_train, y_train)
preds_cm = model_cm.predict(X_test)

cm = confusion_matrix(y_test, preds_cm)
print("Confusion Matrix:")
print(cm)
print(f"True Negatives  (correct no):  {cm[0][0]}")
print(f"False Positives (false alarm):  {cm[0][1]}")
print(f"False Negatives (missed yes):   {cm[1][0]}")
print(f"True Positives  (correct yes):  {cm[1][1]}")


# --- Precision, Recall, F1 ---
# precision = of all predicted positive, how many were actually positive
# recall    = of all actual positives, how many did we catch
# f1        = single score balancing both — best metric for imbalanced data
# in medicine: recall matters most — missing a cancer patient is worse than a false alarm

print(classification_report(y_test, preds_cm, target_names=["Benign", "Malignant"]))

tn, fp, fn, tp = cm.ravel()
precision = tp / (tp + fp)
recall    = tp / (tp + fn)
f1        = 2 * (precision * recall) / (precision + recall)

print(f"Manual Precision: {precision:.4f}")
print(f"Manual Recall:    {recall:.4f}")
print(f"Manual F1:        {f1:.4f}")


# --- ROC Curve and AUC ---
# ROC plots the tradeoff between catching positives and false alarms at every threshold
# AUC summarizes it: 1.0 = perfect | 0.5 = random guessing
# lower threshold = higher recall but more false alarms — can't improve one without the other

probs_roc = model_cm.predict_proba(X_test)[:, 1]
auc       = roc_auc_score(y_test, probs_roc)
print(f"AUC Score: {auc:.4f}")

fpr, tpr, thresholds = roc_curve(y_test, probs_roc)
print("\nSample thresholds and their tradeoffs:")
for i in range(0, len(thresholds), max(1, len(thresholds) // 5)):
    print(f"threshold={thresholds[i]:.2f} | TPR(recall)={tpr[i]:.2f} | FPR(false alarm rate)={fpr[i]:.2f}")


# --- Cross Validation ---
# splits data into K folds, trains K times using a different fold as test each time
# averages the scores — much more reliable than a single train/test split
# low std = model performs consistently regardless of which data it trains on

models_cv = {
    "Decision Tree (depth=3)":  DecisionTreeClassifier(max_depth=3, random_state=42),
    "Decision Tree (depth=10)": DecisionTreeClassifier(max_depth=10, random_state=42),
    "Random Forest":            RandomForestClassifier(n_estimators=100, random_state=42),
}

for name, model in models_cv.items():
    scores = cross_val_score(model, X_cancer, y_cancer, cv=5)
    print(f"{name}")
    print(f"  scores: {scores.round(3)}")
    print(f"  mean: {scores.mean():.4f} | std: {scores.std():.4f}")


# --- Comparing Models ---
# use cross validation + AUC together — when both metrics agree, the choice is clear

candidates = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model",  LogisticRegression(max_iter=10000))
    ]),
    "Decision Tree":  DecisionTreeClassifier(max_depth=3, random_state=42),
    "Random Forest":  RandomForestClassifier(n_estimators=100, random_state=42),
}

print(f"\n{'Model':<25} {'CV Mean':>8} {'CV Std':>8} {'AUC':>8}")
print("-" * 55)

for name, model in candidates.items():
    cv_scores = cross_val_score(model, X_cancer, y_cancer, cv=5)
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]
    auc   = roc_auc_score(y_test, probs)
    print(f"{name:<25} {cv_scores.mean():>8.4f} {cv_scores.std():>8.4f} {auc:>8.4f}")
# logistic regression wins — highest CV mean and AUC on this dataset


# --- Hyperparameter Tuning ---
# GridSearchCV tries every combination in the grid and cross validates each one
# returns the best parameters automatically — eliminates manual guessing

param_grid = {
    "max_depth":         [2, 3, 5, 7, 10],
    "n_estimators":      [50, 100, 200],
    "min_samples_split": [2, 5, 10]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring="roc_auc"
)

warnings.filterwarnings("ignore")
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
preds_best = best_model.predict(X_test)
probs_best = best_model.predict_proba(X_test)[:, 1]

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV AUC:     {grid_search.best_score_:.4f}")
print(f"Test AUC:        {roc_auc_score(y_test, probs_best):.4f}")
print(f"Test Accuracy:   {accuracy_score(y_test, preds_best):.4f}")
print(classification_report(y_test, preds_best, target_names=["Benign", "Malignant"]))

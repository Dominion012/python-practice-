import warnings
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (accuracy_score, roc_auc_score, confusion_matrix,
                             precision_score, recall_score, f1_score,
                             precision_recall_curve, roc_curve, auc)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

# =============================================
# Day 22 - Model Evaluation Deep Dive
# =============================================


# --- Why Accuracy Alone Is Misleading ---
# on a 97% / 3% imbalanced dataset, predicting all-negative scores 97% accuracy
# without detecting a single positive case — accuracy is blind to class imbalance
#
# AUC = probability that a random positive scores higher than a random negative
# AUC 0.80 means: pick any fraud + any non-fraud, model ranks the fraud higher 80% of the time
# AUC is threshold-independent — evaluates ranking quality across all thresholds at once
# AUC 0.5 = random guessing  |  AUC 1.0 = perfect separation

x, y = make_classification(
    n_samples=1000, n_features=10,
    weights=[0.97, 0.03],   # 97% class 0, 3% class 1
    random_state=42
)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

scaler     = StandardScaler()
x_train_sc = scaler.fit_transform(x_train)
x_test_sc  = scaler.transform(x_test)

model = LogisticRegression(max_iter=1000)
model.fit(x_train_sc, y_train)
preds = model.predict(x_test_sc)
probs = model.predict_proba(x_test_sc)[:, 1]   # probabilities for AUC — not binary preds

acc       = accuracy_score(y_test, preds)
auc_score = roc_auc_score(y_test, probs)

dumb     = np.zeros(len(y_test), dtype=int)    # always predicts "not fraud"
dumb_acc = accuracy_score(y_test, dumb)
dumb_auc = roc_auc_score(y_test, dumb)

print(f"Real model — accuracy: {acc:.2f}  AUC: {auc_score:.3f}")
print(f"Dumb model — accuracy: {dumb_acc:.2f}  AUC: {dumb_auc:.3f}")
# accuracy said equal (both 0.96) — AUC caught the dumb model immediately (0.5 = random)


# --- Confusion Matrix ---
# breaks predictions into 4 buckets — more honest than a single accuracy number
#
#              Predicted NO     Predicted YES
# Actual NO  | True Negative  | False Positive |  ← false alarm
# Actual YES | False Negative | True Positive  |  ← missed case
#
# every metric (precision, recall, F1, AUC) is built from these four numbers

cm = confusion_matrix(y_test, preds)
tn, fp, fn, tp = cm.ravel()
print(f"\nTN={tn}  FP={fp}  FN={fn}  TP={tp}")

accuracy_manual  = (tp + tn) / (tp + tn + fp + fn)
precision_manual = tp / (tp + fp) if (tp + fp) > 0 else 0
recall_manual    = tp / (tp + fn)

print(f"Manual — accuracy: {accuracy_manual:.2f}  precision: {precision_manual:.2f}  recall: {recall_manual:.2f}")
# TP=0: model assigns all fraud probabilities below 0.5 — default threshold is wrong for imbalanced data
# AUC=0.80 (model IS learning to rank) but predict() catches nothing at the 0.5 cutoff
print(f"\nFraud probabilities for actual fraud cases: {probs[y_test == 1].round(3)}")
# all below 0.5 — confirms threshold 0.5 is calibrated for balanced data, not 3% fraud


# --- Precision and Recall: the Inescapable Tradeoff ---
# precision = of all I flagged positive, what fraction was real?  → alarm quality
# recall    = of all actual positives, what fraction did I catch? → coverage
#
# lower threshold → more fraud caught (recall ↑) but more false alarms (precision ↓)
# higher threshold → fewer false alarms (precision ↑) but more missed fraud (recall ↓)
# you CANNOT improve both simultaneously — this is a mathematical law, not a model flaw
#
# cancer screening / rescue → maximise recall  (missing a case is catastrophic)
# spam filter on real mail  → maximise precision (blocking real mail is too costly)
# bank fraud                → context-dependent on cost of each error type

print("\nPrecision-Recall tradeoff across thresholds:")
for threshold in [0.05, 0.10, 0.20]:
    p  = (probs >= threshold).astype(int)
    pr = precision_score(y_test, p, zero_division=0)
    re = recall_score(y_test, p)
    f1 = 2 * (pr * re) / (pr + re) if (pr + re) > 0 else 0
    print(f"  threshold={threshold} | precision={pr:.2f} | recall={re:.2f} | f1={f1:.2f} | flagged={p.sum()}")
# false negatives cost the bank money (undetected theft)
# false positives cost customer goodwill (unnecessary calls) — FN is the expensive error


# --- F1 Score: Harmonic Mean, Not Arithmetic Mean ---
# regular average fails: precision=0.01, recall=1.0 → avg=0.505 (looks decent)
# harmonic mean punishes extreme imbalance — same case gets F1=0.02
#
#   F1 = 2 × (precision × recall) / (precision + recall)
#
# F1 peaks at the threshold with the BEST BALANCE — not necessarily highest recall
# use F1 when you want to optimise both precision and recall together

flag_all = np.ones(len(y_test), dtype=int)   # flags everyone — zero discrimination
f1_flag  = f1_score(y_test, flag_all)
avg_flag = (1.0 + y_test.mean()) / 2         # arithmetic avg of precision=1, recall=fraud_rate

print(f"\nFlag-everything model — F1: {f1_flag:.2f}  regular avg would give: {avg_flag:.2f}")
# F1=0.08 vs regular average 0.52 — harmonic mean correctly exposes the useless model


# --- ROC vs Precision-Recall Curve: When AUC Lies ---
# ROC curve: TPR (recall) vs FPR at every threshold
# PR  curve: precision vs recall at every threshold
#
# ROC hides behind True Negatives:
#   FPR = FP / (FP + TN) — with 960 negatives, 96 false alarms gives FPR=0.10 (looks fine)
#   PR has no TN term — every false alarm directly hurts precision, nowhere to hide
#
# random ROC baseline = 0.5 always
# random PR  baseline = fraud_rate (0.03 here) — much lower, honest about difficulty
#
# rule: balanced data → ROC/AUC  |  imbalanced data → PR curve / AUC-PR

fpr_vals, tpr_vals, _ = roc_curve(y_test, probs)
prec_vals, rec_vals, _ = precision_recall_curve(y_test, probs)
roc_auc_val = auc(fpr_vals, tpr_vals)
pr_auc_val  = auc(rec_vals, prec_vals)

print(f"\nROC AUC: {roc_auc_val:.3f}  (random baseline = 0.5)")
print(f"PR  AUC: {pr_auc_val:.3f}  (random baseline = {y_test.mean():.3f})")
# ROC 0.80 sounds good — PR 0.19 tells the honest story about a hard imbalanced problem


# --- Cross Validation: Stratified Folds and Data Leakage ---
# StratifiedKFold preserves class ratio in every fold
# random CV on 3% fraud: some folds may have 0 fraud in test set — evaluation is meaningless
#
# Pipeline prevents data leakage:
#   wrong: scaler.fit_transform(ALL data) before splitting — test stats contaminate training
#   right: Pipeline applies scaler.fit() only inside each training fold automatically
# leakage makes CV scores look better than reality — model fails when deployed

pipe     = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=1000))])
cv_strat = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores   = cross_val_score(pipe, x, y, cv=cv_strat, scoring="roc_auc")

print(f"\nStratified CV AUC scores: {scores.round(3)}")
print(f"Mean: {scores.mean():.3f}  Std: {scores.std():.3f}")
# high std = model is unstable across folds — investigate before deploying
# low std  = model generalises consistently — mean score is trustworthy


# --- Comparing Models: When Metrics Disagree ---
# pick your PRIMARY metric based on the business problem first
# CV mean = main score  |  CV std = tiebreaker (lower = more reliable)
# when metrics disagree: the one matching your business objective wins
# for imbalanced fraud: AUC-PR is the deciding metric — it is the honest one

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
    "Decision Tree":       DecisionTreeClassifier(max_depth=4, random_state=42),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print(f"\n{'Model':<25} {'AUC-ROC':>9} {'AUC-PR':>8} {'F1':>6}")
print("-" * 52)

for name, clf in models.items():
    pipe = Pipeline([("scaler", StandardScaler()), ("model", clf)])
    roc  = cross_val_score(pipe, x, y, cv=cv, scoring="roc_auc").mean()
    pr   = cross_val_score(pipe, x, y, cv=cv, scoring="average_precision").mean()
    f1   = cross_val_score(pipe, x, y, cv=cv, scoring="f1").mean()
    print(f"{name:<25} {roc:>9.3f} {pr:>8.3f} {f1:>6.3f}")

# Random Forest: leads on AUC-ROC and AUC-PR (ranking quality), competitive on F1
# edges the others across 2 of 3 metrics — strongest overall case
# when metrics disagree, the metric matching the business objective decides:
# for imbalanced fraud detection, AUC-PR is always the honest tiebreaker

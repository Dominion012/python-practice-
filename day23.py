import warnings
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

# =============================================
# Day 23 - Bagging vs Boosting, Gradient Boosting & XGBoost
# =============================================


# --- Bagging vs Boosting: Two Philosophies of Ensembling ---
# BAGGING (Day 21's Random Forest):
#   - trees trained in PARALLEL, each on its own bootstrap sample + random feature subset
#   - final prediction = average / majority vote
#   - reduces VARIANCE (σ²/N) — doesn't fix a bias every tree shares
#
# BOOSTING:
#   - trees trained SEQUENTIALLY — each new tree targets the ERRORS of the current ensemble
#   - final prediction = weighted SUM of all trees, not a vote
#   - reduces BIAS — directly attacks the mistakes the ensemble is still making
#   - risk: too many rounds / too aggressive learning rate -> starts fitting noise (overfits)
#
# bagging averages many ok-ish independent guesses;
# boosting builds a chain where each link's only job is to fix the previous link's mistake


# --- Manual Gradient Boosting: Fitting Residuals with Shrinkage ---
# round 0: predict the mean for everyone
# each round after that:
#   1. residuals = actual - current prediction        ("what's left to explain")
#   2. fit a small tree to PREDICT THOSE RESIDUALS
#   3. add a SHRUNK version of that tree's prediction to the running total
#
#   F(x) = F0(x) + lr*tree1(x) + lr*tree2(x) + ... + lr*treeN(x)
#
# learning_rate=0.5 means each tree only corrects HALF of what it found —
# this "shrinkage" is deliberate: small steps generalise better than big jumps

x_toy = np.array([[1], [2], [3], [4], [5], [6], [7], [8]], dtype=float)
y_toy = np.array([2, 4, 5, 4, 5, 7, 8, 9], dtype=float)

learning_rate = 0.5
prediction = np.full(len(y_toy), y_toy.mean())   # round 0: everyone predicts the mean

print("--- Manual Gradient Boosting ---")
print(f"Round 0 (mean only) | SSR: {np.sum((y_toy - prediction) ** 2):.2f}")

for round_num in range(1, 4):
    residuals = y_toy - prediction
    stump = DecisionTreeRegressor(max_depth=1, random_state=42)
    stump.fit(x_toy, residuals)
    prediction = prediction + learning_rate * stump.predict(x_toy)
    ssr = np.sum((y_toy - prediction) ** 2)
    print(f"Round {round_num} (+1 stump)    | SSR: {ssr:.2f}")

# SSR drops every round (38.00 -> 15.50 -> 9.02 -> ...) but by SHRINKING amounts —
# diminishing returns. the big, easy patterns get captured first; what's left for
# later rounds is smaller and noisier. real gradient boosting exploits this by using
# hundreds of TINY rounds (lr=0.01-0.1) instead of a few big ones.


# --- Gradient Boosting -> XGBoost ---
# the loop above IS gradient boosting. sklearn's GradientBoostingClassifier/Regressor
# formalises it. XGBoost ("Extreme Gradient Boosting") is the same additive idea,
# engineered hard:
#   - L1/L2 regularization on leaf weights (penalises overly confident leaves)
#   - handles missing values natively (learns a default split direction)
#   - histogram-based split finding -> much faster on large datasets
#   - built-in early stopping (stop adding trees once validation score stalls)
#
# key hyperparameters and the tradeoff they control:
#   n_estimators   -> how many trees (rounds) get added
#   learning_rate  -> how much each tree's correction counts (shrinkage)
#   max_depth      -> how complex each individual tree is allowed to be
#
# more/shallower trees + lower lr -> trains slower, usually generalises BEST
# deep trees + many rounds        -> classic boosting overfit: memorises training noise
#
# NOTE: XGBoost (`pip install xgboost`) needs the `libomp` system library on macOS,
# which requires Homebrew (not set up on this machine yet). XGBClassifier follows the
# exact same fit/predict API as GradientBoostingClassifier below — once libomp is
# installed, it can be dropped straight into the `models` dict with no other changes.


# --- Model Comparison: LR vs Random Forest vs Gradient Boosting ---
# same imbalanced fraud-style dataset as Day 22 (97% / 3% classes)

x, y = make_classification(
    n_samples=1000, n_features=10,
    weights=[0.97, 0.03],
    random_state=42
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting":   GradientBoostingClassifier(random_state=42),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print(f"\n{'Model':<22} {'AUC-ROC':>9} {'AUC-PR':>8} {'F1':>6}")
print("-" * 48)

for name, clf in models.items():
    pipe = Pipeline([("scaler", StandardScaler()), ("model", clf)])
    roc  = cross_val_score(pipe, x, y, cv=cv, scoring="roc_auc").mean()
    pr   = cross_val_score(pipe, x, y, cv=cv, scoring="average_precision").mean()
    f1   = cross_val_score(pipe, x, y, cv=cv, scoring="f1").mean()
    print(f"{name:<22} {roc:>9.3f} {pr:>8.3f} {f1:>6.3f}")

# both ensembles beat LR by a wide margin -> this dataset has a non-linear boundary
# that a straight-line model can't capture
# Gradient Boosting edges out Random Forest on AUC-ROC and especially F1 — its
# "sequentially fix what's still wrong" approach squeezes out gains specifically on
# the HARD cases, which on an imbalanced dataset are disproportionately the fraud ones


# --- AUC-PR: Average, Not a Single Operating Point ---
# AUC-PR (average precision) = a weighted average of precision across EVERY threshold
# on the precision-recall curve — not the precision you'd see at the one threshold
# you actually deploy (that still requires picking a cutoff and checking precision_score,
# like Day 22's threshold scan)
#
# the number that matters is AUC-PR vs its BASELINE:
#   AUC-ROC random baseline = 0.5, always
#   AUC-PR  random baseline = the positive class rate itself (0.03 here)
# AUC-PR ~0.6 vs a 0.03 baseline -> roughly 20x better than random at ranking
# fraud above non-fraud — that comparison is the headline, not the raw 0.6


# --- Choosing Between Logistic Regression, Random Forest, and (Gradient) Boosting ---
#
# LOGISTIC REGRESSION
#   + fastest to train, instant to predict, coefficients are directly explainable
#   + great baseline — if it's already good, you may not need anything fancier
#   - assumes a roughly linear decision boundary (in log-odds space)
#
# RANDOM FOREST (bagging)
#   + trains in PARALLEL — all trees independent, can use every CPU core at once
#   + ~one knob that matters (n_estimators) — hard to make it badly wrong
#   - many trees -> slower predictions, "feature importance" only, not per-prediction reasoning
#
# GRADIENT BOOSTING / XGBOOST (boosting)
#   + usually the highest accuracy on structured/tabular data
#   + regularization + early stopping give fine control over the bias/variance line
#   - trains SEQUENTIALLY — tree N depends on tree N-1's output, can't parallelise across trees
#   - 3 interacting knobs (lr, n_estimators, max_depth) — easy to overfit if mistuned
#
# practical default: ship Random Forest first (robust, low effort, safe).
# invest in tuning Gradient Boosting/XGBoost only if the extra accuracy is worth the
# slower training and added tuning risk.

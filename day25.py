import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pandas as pd

# =============================================
# Day 25 - Capstone: Customer Churn — Tuning Metrics, Deployment & Feature Importance
#   Steps 1-5: synthetic churn dataset, feature engineering, baseline model comparison
#   Steps 6-9: GridSearchCV scoring choice, the final deployment pipeline,
#              scoring brand-new customers, coef_ vs permutation importance
# =============================================

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting":    GradientBoostingClassifier(random_state=42),
}
features = ["tenure_months", "monthly_charge", "support_calls", "usage_hours", "age",
            "num_addons", "total_spent", "is_new_customer", "usage_per_charge", "is_month_to_month"]





np.random.seed(42)
n = 200

tenure_months = np.random.randint(1,73, n)
monthly_charge = np.random.uniform(20,120, n)
support_calls = np.random.poisson(1.5,n)
usage_hours = np.random.normal(20,8,n)
age = np.random.randint(18,71,n)
contract_type = np.random.choice(["month-to-month", "one_year", "two_year"], n , p = [0.5, 0.3,0.2])
num_addon_services = np.random.randint(0, 4,n)


z_tenure = (tenure_months - tenure_months.mean()) / tenure_months.std()
z_charge = (monthly_charge - monthly_charge.mean()) / monthly_charge.std()
z_calls = (support_calls - support_calls.mean()) / support_calls.std()
z_usage = (usage_hours - usage_hours.mean()) / usage_hours.std()
z_addons = (num_addon_services - num_addon_services.mean()) / num_addon_services.std()
is_month_to_month = (contract_type == "month-to-month").astype(int)


dt = pd.DataFrame({
    "tenure_months" : tenure_months,
    "monthly_charge" : monthly_charge,
    "support_calls"   :support_calls,
    "usage_hours"    : usage_hours,
    "age"      : age,
    "contract_type" : contract_type,
    "num_addons" :    num_addon_services

}
)
b0       = -0.5
b_tenure = -0.8
b_charge =  0.5
b_calls  =  0.6
b_usage  = -0.5
b_mtm    =  0.7
b_addons = -0.4



log_odds = (b0 + b_tenure*z_tenure + b_charge*z_charge + b_calls*z_calls
             + b_usage*z_usage + b_mtm*is_month_to_month + b_addons*z_addons)
p = 1 / (1 + np.exp(-log_odds))
churn = np.random.binomial(1, p)
dt["churn"] = churn
dt["total_spent"] = dt["monthly_charge"] * dt["tenure_months"]
dt["usage_hours"] = dt["usage_hours"].clip(lower=0)
dt["is_new_customer"] = (dt["tenure_months"] <= 6).astype(int)
dt["usage_per_charge"] = dt["usage_hours"] / dt["monthly_charge"]
dt["is_month_to_month"] = is_month_to_month


# --- Steps 1-3: Synthetic dataset + feature engineering ---
# churn is generated from a TRUE logistic formula over z-scored raw features
# (tenure, charge, support calls, usage, contract type, addons) — only those
# 6 are real causal drivers. age is generated independently and never enters
# the formula (foreshadows Step 9: age turns out to be pure noise).
# 4 engineered columns are added on top: total_spent, is_new_customer,
# usage_per_charge, is_month_to_month — some of these duplicate information
# already present in the raw features (foreshadows Step 9 again).
X = dt[features]
y = dt["churn"]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# --- Steps 4-5: Baseline model comparison (5-fold stratified CV) ---
# Logistic Regression comes out on top across all 3 metrics here — this
# dataset's churn formula IS a logistic model, so a linear decision boundary
# is the right fit. Random Forest and Gradient Boosting can't beat a model
# that matches the true data-generating process.
print(f"\n{'Model':<22} {'AUC-ROC':>9} {'AUC-PR':>8} {'F1':>6}")
print("-" * 48)

for name, clf in models.items():
    pipe = Pipeline([("scaler", StandardScaler()), ("model", clf)])
    roc  = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc").mean()
    pr   = cross_val_score(pipe, X, y, cv=cv, scoring="average_precision").mean()
    f1   = cross_val_score(pipe, X, y, cv=cv, scoring="f1").mean()
    print(f"{name:<22} {roc:>9.3f} {pr:>8.3f} {f1:>6.3f}")

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

from sklearn.model_selection import GridSearchCV

pipe_lr = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=1000, solver="liblinear"))])

param_grid = {
    "model__C": [0.01, 0.1, 1, 10, 100],
    "model__penalty": ["l1", "l2"]
}

grid = GridSearchCV(pipe_lr, param_grid, cv=cv, scoring="f1")


grid.fit(X_train, y_train)
best_params = grid.best_params_
best_score = grid.best_score_
best_pipe = grid.best_estimator_

print(f"best_params: {best_params}")
print(f"best_score (CV F1): {best_score:.3f}")
predict = grid.predict(X_test)
probab = best_pipe.predict_proba(X_test)[:, 1]
print(f"Tuned LogReg precision={precision_score(y_test, predict):.2f}  "
      f"recall={recall_score(y_test, predict):.2f}  "
      f"f1={f1_score(y_test, predict):.2f}  "
      f"auc={roc_auc_score(y_test, probab):.3f}")

# --- Step 6: scoring= picks the hyperparameters — and a mismatched metric can
# pick a model that's WORSE at the thing you'll actually deploy with ---
#
# First pass used scoring="roc_auc": best_params = {C: 0.01, penalty: 'l2'}
#   CV AUC: 0.813 -> 0.841 (genuinely better ranking)
#   but test F1 dropped 0.67 -> 0.53 (recall 0.65 -> 0.47)
#
# C=0.01 = strong L2 regularization -> shrinks ALL coefficients toward zero ->
# every probability gets pulled toward 0.5 (the model hedges on everyone).
# Relative ORDER barely changes (AUC holds/improves) but several borderline
# customers cross from just-above-0.5 to just-below-0.5 -> predict.sum()
# dropped from 17 (true churners) to 13.
#
# Switched to scoring="f1": best_params = {C: 10, penalty: 'l2'} (CV F1=0.709).
# C=10 = weak regularization -> probabilities spread out (0.02-0.99 instead of
# 0.33-0.70) -> predict.sum()=16, matching y_test.sum()=17 -> test F1 back to
# 0.67, AUC barely moved (0.744 vs 0.749).
#
# LESSON: tune on roc_auc if you're RANKING customers for a human to triage;
# tune on f1/precision/recall if the deployed code calls .predict() at 0.5.
# Tuning on the wrong one can push C in the OPPOSITE direction from what
# .predict() needs.


# --- Step 7: Final deployable pipeline — and a leakage trap hiding in plain sight ---
# best_pipe = grid.best_estimator_ does NOT copy the model — best_pipe and
# grid.best_estimator_ are the SAME object. Calling best_pipe.fit(X, y) BEFORE
# the Step 6 test-set evaluation above would mutate that shared object in
# place, so grid.predict(X_test)/best_pipe.predict_proba(X_test) would then run
# on a model that already SAW X_test during training (test AUC jumped
# 0.744 -> 0.785, F1 dropped 0.67 -> 0.62 — both numbers became dishonest).
#
# FIX: order matters. Finish ALL test-set evaluation first (done above), THEN
# refit on the full dataset (X, y) for deployment — the all-data refit is a
# one-way trip, there's no "undo" once X_test has been folded back into training.
best_pipe.fit(X, y)
print("Final model coefficients:", best_pipe.named_steps["model"].coef_)


# --- Step 8: Scoring brand-new customers ---
# the deployed pipeline expects RAW feature values (the scaler inside it
# already learned mean/std from .fit(X, y) above) — AND it expects all 4
# engineered columns (total_spent, is_new_customer, usage_per_charge,
# is_month_to_month) computed by hand for the new row, the exact same way
# they were computed for `dt`.
new_customer = pd.DataFrame({
    "tenure_months": [2],
    "monthly_charge": [95.0],
    "support_calls": [4],
    "usage_hours": [10.0],
    "age": [29],
    "num_addons": [1],
    "is_month_to_month": [1],
})
new_customer["total_spent"] = new_customer["monthly_charge"] * new_customer["tenure_months"]
new_customer["is_new_customer"] = (new_customer["tenure_months"] <= 6).astype(int)
new_customer["usage_per_charge"] = new_customer["usage_hours"] / new_customer["monthly_charge"]

new_customer = new_customer[features]  # match column order

pred = best_pipe.predict(new_customer)
prob = best_pipe.predict_proba(new_customer)[:, 1]
print(f"Predicted churn: {pred[0]}  (probability: {prob[0]:.2f})")

customer_2 = pd.DataFrame({
    "tenure_months": [24],
    "monthly_charge": [95.0],
    "support_calls": [1],
    "usage_hours": [10.0],
    "age": [29],
    "num_addons": [1],
    "is_month_to_month": [0]
})
customer_2["total_spent"] = customer_2["monthly_charge"] * customer_2["tenure_months"]
customer_2["is_new_customer"] = (customer_2["tenure_months"] <= 6).astype(int)
customer_2["usage_per_charge"] = customer_2["usage_hours"] / customer_2["monthly_charge"]

customer_2 = customer_2[features]
pred_2 = best_pipe.predict(customer_2)
prob_2 = best_pipe.predict_proba(customer_2)[:,1]
print(f"Predicted churn: {pred_2[0]}  (probability: {prob_2[0]:.2f})")
# Customer 1 (tenure=2mo, 4 support calls, month-to-month): churn=1, prob=0.99
# Customer 2 (tenure=24mo, 1 support call, locked-in contract): churn=1, prob=0.78
# -> direction is right (risk dropped) but smaller than expected -> Step 9 explains why


# --- Step 9: coef_ vs permutation_importance — multicollinearity exposed ---
# Breaking customer_2's prediction into per-feature log-odds contributions
# showed the drop was driven almost entirely by 3 features (support_calls,
# tenure_months, is_month_to_month) — but PARTIALLY OFFSET by total_spent and
# is_new_customer moving the WRONG direction.
#
# permutation_importance (shuffle each column, measure the F1 drop) explains
# why — sorted by real importance:
#   tenure_months      coef=-1.542  importance=0.1523  <- dominant real signal
#   support_calls      coef= 0.877  importance=0.0481  <- real signal
#   usage_per_charge   coef=-0.552  importance=0.0279  <- real signal (engineered!)
#   is_month_to_month  coef= 0.505  importance=0.0200  <- real signal
#   monthly_charge     coef= 0.207  importance~0       <- redundant
#   num_addons         coef=-0.520  importance~0       <- redundant
#   total_spent        coef= 0.277  importance~0       <- redundant (= charge*tenure)
#   is_new_customer    coef=-0.176  importance~0       <- redundant (= tenure<=6)
#   usage_hours        coef~0       importance~0       <- redundant
#   age                coef=-0.184  importance=-0.017  <- pure noise (never in
#                                                          the true churn formula)
#
# total_spent and is_new_customer are deterministic FUNCTIONS of tenure_months
# already in the model -> the model can't cleanly attribute "the tenure effect"
# to one feature, so it splits/leaks weight onto the redundant copies, and
# whatever's left over can land with either sign — that's why their coef_
# signs looked backwards.
#
# LESSON: coef_ reflects how the model internally balanced its books under
# collinearity — NOT what's actually driving predictions. permutation_importance
# measures the real-world impact of removing each information source. Only 4 of
# 10 features here carry genuine signal; the rest (including a feature with a
# "big-looking" coefficient, total_spent) could be dropped with near-zero cost.
from sklearn.inspection import permutation_importance

perm = permutation_importance(best_pipe, X, y, scoring="f1", n_repeats=10, random_state=42)
coefs = best_pipe.named_steps["model"].coef_[0]

print(f"\n{'Feature':<18} {'Coefficient':>11} {'Perm. Importance':>17}")
print("-" * 48)
for f, c, imp in zip(features, coefs, perm.importances_mean):
    print(f"{f:<18} {c:>11.3f} {imp:>17.4f}")


import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# =============================================
# Day 21 - Classification Deep Dive
# =============================================


# --- Why Logistic Regression, Not Linear Regression ---
# linear regression can output 1.4 or -0.2 — meaningless as a probability
# we need a function that: always outputs 0-1, and saturates near certainty at the extremes
# that function is the sigmoid — and it comes from modelling log-odds with a straight line:
#
#   log( p / (1-p) )  =  b0 + b1*x1 + b2*x2 + ...
#
# p/(1-p) = odds   |   log(odds) = logit   |   sigmoid = algebraic inverse of logit
# chain: linear score → log-odds → undo with exp → odds → divide → probability (0-1)

b0    = -4
b1    = 1.2
hours = 4

log_odds    = b0 + b1 * hours
odds        = np.exp(log_odds)
probability = odds / (1 + odds)   # same as sigmoid: 1 / (1 + e^-score)

print(f"log-odds: {log_odds}")
print(f"odds:     {odds:.2f}  →  {odds:.2f}-to-1 in favour of passing")
print(f"prob:     {probability:.2f}")


# --- Coefficients as Odds Multipliers ---
# raw coefficient b1=1.2 is an abstract weight — meaningless to a non-technical person
# e^coef = how much each unit increase in that feature MULTIPLIES the odds
# "each extra hour of studying multiplies the odds of passing by e^1.2 ≈ 3.32x"
# always report e^coef to stakeholders, not the raw coefficient

multiplier = np.exp(b1)
print(f"\nEach extra hour multiplies odds of passing by {multiplier:.2f}x")


# --- Spam Classifier with Odds Ratio Interpretation ---
# after fitting logistic regression, convert every coefficient with e^coef
# this turns the black-box weight into a human-readable business insight
# "chances" vs "odds": e^coef is an ODDS multiplier — not interchangeable with probability

data_spam = pd.DataFrame({
    "word_free":   [3, 0, 5, 1, 0, 4, 0, 2, 0, 6, 1, 0, 3, 0, 2],
    "word_winner": [2, 0, 3, 0, 0, 2, 0, 1, 0, 4, 0, 0, 2, 0, 1],
    "num_links":   [5, 1, 8, 2, 0, 6, 1, 3, 0, 9, 2, 1, 4, 0, 3],
    "all_caps":    [1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0],
    "num_exclaim": [4, 0, 6, 1, 0, 5, 0, 2, 0, 7, 1, 0, 3, 0, 2],
    "spam":        [1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1]
})

X_spam = data_spam[["word_free", "word_winner", "num_links", "all_caps", "num_exclaim"]]
y_spam = data_spam["spam"]

model_spam = LogisticRegression()
model_spam.fit(X_spam, y_spam)

coef_spam = pd.Series(model_spam.coef_[0], index=X_spam.columns)
print("\nSpam feature odds multipliers (sorted strongest → weakest):")
for name, val in coef_spam.sort_values(ascending=False).items():
    print(f"  {name:<15} {np.exp(val):.3f}x")
# num_links = strongest spam signal — each extra link roughly doubles spam odds


# --- Decision Trees: How Splits Are Chosen ---
# at every node the tree tries every possible split on every feature
# picks the one that makes children most "pure" — measured with Gini impurity:
#
#   Gini = 1 - Σ(pᵢ²)   where pᵢ = fraction of class i in that group
#
# perfectly pure group: Gini = 0   |   50/50 mix: Gini = 0.5 (worst)
# information gain = parent Gini − weighted average of children's Gini values
# tree greedily picks the split with the highest information gain at each node

def gini(x, y):
    total  = x + y
    px, py = x / total, y / total
    return 1 - (px**2 + py**2)

parent       = gini(6, 4)   # parent node: 6 spam, 4 not-spam
left         = gini(1, 4)   # left child after split: 1 spam, 4 not-spam
right        = gini(5, 0)   # right child: 5 spam, 0 not-spam — perfectly pure

weighted_avg     = (5/10) * left + (5/10) * right
information_gain = parent - weighted_avg

print(f"\nGini — parent: {parent:.2f}  left: {left:.2f}  right: {right:.2f}")
print(f"Weighted avg children: {weighted_avg:.2f}")
print(f"Information gain:      {information_gain:.2f}")

# right = 0.0 → perfectly pure → this split is ideal
# operator precedence trap: p/10**2 means p/100, not (p/10)² — always use parentheses


# --- Random Forest: Why Averaging Beats One Tree ---
# one deep tree: LOW BIAS (fits anything) but HIGH VARIANCE (changes with every dataset)
# random forest fights variance with:
#   1. bagging — each tree trains on a random bootstrap sample of the data
#   2. random feature subsets — each split considers a random subset of features
# → decorrelates the trees so they fail independently, not all together
#
# variance of the mean of N independent estimates = σ²/N
# more (sufficiently different) trees → variance shrinks → predictions stabilise
# same principle as: one poll of 10 people is noisy, average of 100 polls is reliable

data_dt = pd.DataFrame({
    "age":     [22, 45, 31, 60, 25, 52, 38, 29, 55, 41, 33, 48, 27, 65, 36],
    "balance": [1200, 8000, 3000, 15000, 500, 12000, 6000, 2000, 11000, 7000, 4000, 9000, 800, 18000, 5000],
    "job":     [0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0],
    "default": [1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0]
})

X_dt = data_dt[["age", "balance", "job"]]
y_dt = data_dt["default"]

cv_tree   = cross_val_score(DecisionTreeClassifier(max_depth=None), X_dt, y_dt, cv=5)
cv_forest = cross_val_score(RandomForestClassifier(n_estimators=100, random_state=42), X_dt, y_dt, cv=5)

print(f"\nDecision Tree  — mean: {cv_tree.mean():.3f}  std: {cv_tree.std():.3f}")
print(f"Random Forest  — mean: {cv_forest.mean():.3f}  std: {cv_forest.std():.3f}")

# on tiny clean data this may not show clearly — the bias-variance story holds at scale
# perfect scores (1.0, std=0.0) on small data = red flag, not proof of a great model


# --- KNN: Distance, Scaling, and the K Tradeoff ---
# no model is built — memorises all training data, finds K nearest neighbours at predict time
# "nearest" = Euclidean distance → scaling is REQUIRED
# without StandardScaler, large-range features dominate distance (balance 0-18000 vs age 22-65)
#
# K controls the bias-variance tradeoff directly:
#   small K → low bias, HIGH variance — one odd neighbour swings the vote
#   large K → HIGH bias, low variance — "majority class always wins", ignores local patterns

scaler  = StandardScaler()
X_dt_sc = scaler.fit_transform(X_dt)

print("\nKNN — effect of K:")
for k in [1, 3, 5, 7, 9, 11]:
    score = cross_val_score(KNeighborsClassifier(n_neighbors=k), X_dt_sc, y_dt, cv=5).mean()
    print(f"  k={k:>2}: {score:.3f}")
# large K approaches "predict majority class regardless of input" — pure underfitting


# --- How to Choose a Classifier ---
# 1. need to explain the prediction?         → logistic regression or decision tree
# 2. boundary is likely a straight line?     → logistic regression
# 3. complex non-linear feature interactions? → decision tree / random forest
# 4. large noisy dataset?                    → random forest (reliable default)
# 5. prediction speed matters at runtime?    → logistic regression / tree (instant)
#                                            → KNN is slow: scans entire training set per query
# when unsure: start with logistic regression — underperformance tells you the boundary is non-linear


# --- Threshold Tuning: a Business Decision, Not a Math Problem ---
# .predict() applies a 0.5 cutoff by default — this is arbitrary, not a law
# move the threshold based on the COST of each error type:
#
#   false positive = false alarm     |   false negative = missed real case
#
# rescue / cancer screening  → low threshold (0.3)  — missing a case is catastrophic
# spam filter on real mail   → high threshold (0.7)  — false alarms are too costly
#
# two thresholds can have identical ACCURACY but very different FP/FN counts
# accuracy can never tell you which is better — you need to look at FP and FN separately

probs_demo = np.array([0.82, 0.11, 0.45, 0.67, 0.29, 0.91, 0.38, 0.55, 0.14, 0.73])
y_demo     = np.array([1,    0,    1,    1,    0,    1,    0,    1,    0,    1   ])

print("\nThreshold scan:")
print(f"{'Threshold':>10} {'Accuracy':>10} {'Predicted+':>12} {'FP':>5} {'FN':>5}")
print("-" * 46)
for t in [0.3, 0.5, 0.7]:
    preds = (probs_demo >= t).astype(int)
    fp    = ((preds == 1) & (y_demo == 0)).sum()
    fn    = ((preds == 0) & (y_demo == 1)).sum()
    print(f"{t:>10.1f} {accuracy_score(y_demo, preds):>10.2f} {preds.sum():>12} {fp:>5} {fn:>5}")

# 0.3 and 0.5 have the same accuracy but different error profiles
# for rescue: 0.3 — zero false negatives (no survivor left behind) outweighs the extra false alarm

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif

# =============================================
# Day 24 - Capstone: Pipelines for Feature Selection + Hyperparameter Tuning
# =============================================


# --- Why Feature Selection Belongs INSIDE the Pipeline ---
# same leakage rule as Day 22's StandardScaler:
#   wrong: run SelectKBest on the FULL dataset, then cross-validate the model
#          -> the "best k features" choice was informed by data the model is later tested on
#   right: SelectKBest as a Pipeline step -> GridSearchCV refits it on ONLY each
#          training fold, so feature selection never sees that fold's test rows
#
# nested param names use "step__param":
#   "select__k"          -> the k inside the SelectKBest step
#   "model__n_estimators" -> the n_estimators inside the RandomForest step
# tuning both together matters because the best k depends on which model will use
# those features, and the best model hyperparameters depend on which features survive


# --- Titanic Feature Engineering (from Day 20) ---
df = pd.read_csv("tested.csv")
df = df.drop(columns=["Cabin", "Name", "Ticket", "PassengerId"])
df["Age"]  = df["Age"].fillna(df["Age"].median())
df["Fare"] = df["Fare"].fillna(df["Fare"].median())
df["Sex"]      = (df["Sex"] == "female").astype(int)
df["Embarked"] = df["Embarked"].fillna("S")
df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})
df["FamilySize"] = df["SibSp"] + df["Parch"]
df["IsAlone"]    = (df["FamilySize"] == 0).astype(int)

features = ["Pclass", "Sex", "Age", "Fare", "SibSp", "Parch", "Embarked", "FamilySize", "IsAlone"]
x_titanic = df[features]
y_titanic = df["Survived"]


# --- Pipeline + GridSearchCV: Tune select__k and model hyperparameters together ---
pipe_s = Pipeline([
    ("scaler", StandardScaler()),
    ("select", SelectKBest(score_func=f_classif, k=5)),
    ("model", RandomForestClassifier(random_state=42))
])

params_grid = {
    "select__k": [4, 6, 8],
    "model__n_estimators": [100, 200],
    "model__max_depth": [3, 5, None]
}

grids_search = GridSearchCV(pipe_s, params_grid, cv=5, scoring="roc_auc")
grids_search.fit(x_titanic, y_titanic)

bests_pipe  = grids_search.best_estimator_
bests_param = grids_search.best_params_
bests_score = grids_search.best_score_

print("Best params:", bests_param)
print("Selected features:", bests_pipe.named_steps["select"].get_support())
# -> k=4 won; mask keeps Fare, Parch, FamilySize, IsAlone
# -> Pclass, Sex, Age, SibSp, Embarked were dropped


# --- The f_classif Trap: When the "Perfect" Feature Scores Worst ---
# tested.csv is the Kaggle Titanic TEST set with "Survived" grafted on from the
# trivial "all women survive, all men die" baseline submission -> Sex correlates
# with Survived at 1.0, a PERFECT predictor
#
# f_classif's F-statistic = (between-group variance) / (within-group variance)
# perfect separation -> zero within-group variance -> division by zero ->
# garbage score instead of "infinitely good"

print(bests_pipe.named_steps["select"].scores_)
# Sex's score: -8.99e+16 (with p=nan) -- a NEGATIVE F-statistic, which is
# mathematically impossible (F is always >= 0). SelectKBest keeps the HIGHEST
# scores, so this broken value sorted Sex dead last and dropped it -- exactly
# backwards from reality.
#
# LESSON: an automated tool's output (get_support(), best_params_, etc.) can look
# completely normal while being silently wrong underneath. Before trusting a
# feature-selection mask, sanity-check the raw scores for impossible values
# (negative F-stats, NaNs) -- don't just read the final True/False mask.

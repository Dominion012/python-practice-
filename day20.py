import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# =============================================
# Day 20 - End-to-End Machine Learning Project
# Titanic Survival Predictor
# =============================================


# ============================================
# Step 1 — Load and Explore
# ============================================
# understand shape, columns, missing values, and class balance before touching any model

df = pd.read_csv("tested.csv")

print("Shape:", df.shape)
print("\nMissing values per column:")
print(df.isnull().sum())
print(f"\nSurvival rate: {df['Survived'].mean():.2%}")
print(df["Survived"].value_counts())


# ============================================
# Step 2 — Clean the Data
# ============================================
# drop Cabin (78% missing — unusable)
# drop Name, Ticket, PassengerId — unique identifiers, no learnable pattern
# fill Age and Fare with median

df = df.drop(columns=["Cabin", "Name", "Ticket", "PassengerId"])
df["Age"]  = df["Age"].fillna(df["Age"].median())
df["Fare"] = df["Fare"].fillna(df["Fare"].median())

print("\nMissing values after cleaning:")
print(df.isnull().sum())


# ============================================
# Step 3 — Feature Engineering
# ============================================
# models need numbers — convert text columns to integers
# FamilySize and IsAlone: new features that may improve predictions

df["Sex"]      = (df["Sex"] == "female").astype(int)       # female=1, male=0
df["Embarked"] = df["Embarked"].fillna("S")
df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})

df["FamilySize"] = df["SibSp"] + df["Parch"]
df["IsAlone"]    = (df["FamilySize"] == 0).astype(int)     # 1=traveling alone

print("\nSample engineered features:")
print(df[["Sex", "Embarked", "FamilySize", "IsAlone"]].head())


# ============================================
# Step 4 — Train / Test Split
# ============================================
# 80% train, 20% test — check survival rates match to confirm representative split

features = ["Pclass", "Sex", "Age", "Fare", "SibSp", "Parch",
            "Embarked", "FamilySize", "IsAlone"]

X = df[features].fillna(df[features].median())
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining set: {X_train.shape[0]} passengers")
print(f"Test set:     {X_test.shape[0]} passengers")
print(f"Survival rate — train: {y_train.mean():.2%} | test: {y_test.mean():.2%}")


# ============================================
# Step 5 — Train Multiple Models
# ============================================
# KNN requires scaling (distance-based) — LR and RF do not

scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

models = {
    "Logistic Regression": (LogisticRegression(max_iter=1000), False),
    "Random Forest":       (RandomForestClassifier(n_estimators=100, random_state=42), False),
    "KNN":                 (KNeighborsClassifier(n_neighbors=5), True),
}

trained = {}
for name, (model, needs_scale) in models.items():
    X_tr = X_train_sc if needs_scale else X_train
    model.fit(X_tr, y_train)
    trained[name] = (model, needs_scale)
    print(f"{name} trained.")


# ============================================
# Step 6 — Evaluate Each Model
# ============================================
# accuracy, AUC, CV mean, confusion matrix, precision/recall/F1

print(f"\n{'='*55}")
print(f"{'MODEL EVALUATION':^55}")
print(f"{'='*55}")

results = {}

for name, (model, needs_scale) in trained.items():
    X_te  = X_test_sc if needs_scale else X_test
    preds = model.predict(X_te)
    probs = model.predict_proba(X_te)[:, 1]

    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)
    cv  = cross_val_score(model,
                          X_train_sc if needs_scale else X_train,
                          y_train, cv=5).mean()

    results[name] = {"accuracy": acc, "auc": auc, "cv_mean": cv}

    print(f"\n--- {name} ---")
    print(f"Accuracy: {acc:.4f} | AUC: {auc:.4f} | CV Mean: {cv:.4f}")
    print(confusion_matrix(y_test, preds))
    print(classification_report(y_test, preds, target_names=["Died", "Survived"]))


# ============================================
# Step 7 — Compare and Pick the Best
# ============================================
# when CV mean and AUC agree → confident choice
# tiebreaker: CV mean (most reliable — tested across all folds)

print(f"\n{'='*55}")
print(f"{'MODEL COMPARISON':^55}")
print(f"{'='*55}")
print(f"\n{'Model':<25} {'Accuracy':>9} {'AUC':>8} {'CV Mean':>9}")
print("-" * 55)

for name, scores in results.items():
    print(f"{name:<25} {scores['accuracy']:>9.4f} {scores['auc']:>8.4f} {scores['cv_mean']:>9.4f}")

best_name = max(results, key=lambda x: results[x]["auc"])
print(f"\nBest model: {best_name}")


# ============================================
# Step 8 — Predict on New Passengers
# ============================================

best_model = trained[best_name][0]

new_passengers = pd.DataFrame({
    "Pclass":     [1,    3,    2,    1   ],
    "Sex":        [1,    0,    1,    0   ],   # 1=female, 0=male
    "Age":        [28,   22,   35,   55  ],
    "Fare":       [80,   7.5,  21,   60  ],
    "SibSp":      [1,    0,    1,    0   ],
    "Parch":      [0,    0,    2,    0   ],
    "Embarked":   [1,    0,    0,    0   ],   # 0=S, 1=C, 2=Q
    "FamilySize": [1,    0,    3,    0   ],
    "IsAlone":    [0,    1,    0,    1   ]
})

preds_new = best_model.predict(new_passengers)
probs_new = best_model.predict_proba(new_passengers)[:, 1]

profiles = [
    "1st class female, age 28, with husband",
    "3rd class male,   age 22, alone",
    "2nd class female, age 35, with family",
    "1st class male,   age 55, alone"
]

print(f"\n{'='*55}")
print(f"{'NEW PASSENGER PREDICTIONS':^55}")
print(f"{'='*55}")
for profile, pred, prob in zip(profiles, preds_new, probs_new):
    outcome = "SURVIVED" if pred == 1 else "DIED"
    print(f"{profile}")
    print(f"  → {outcome} ({prob:.0%} probability)\n")


# ============================================
# Step 9 — What the Model Learned
# ============================================
# positive coefficient = increases survival chance
# negative coefficient = decreases survival chance

lr_model    = trained["Logistic Regression"][0]
coef        = pd.Series(lr_model.coef_[0], index=features)
coef_sorted = coef.sort_values()

print(f"\n{'='*55}")
print(f"{'FEATURE IMPORTANCE (Logistic Regression)':^55}")
print(f"{'='*55}")
print(coef_sorted)
print(f"\nStrongest survival signal: {coef_sorted.idxmax()}")   # Sex (female)
print(f"Strongest death signal:    {coef_sorted.idxmin()}")    # IsAlone

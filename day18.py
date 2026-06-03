import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# =============================================
# Day 18 - Classification Models Deep Dive
# =============================================


# --- Classification vs Regression ---
# regression predicts a number (price, score, salary)
# classification predicts a category (spam/not spam, survived/died, pass/fail)
# binary classification = 2 labels   |   multi-class = 3+ labels


# --- Logistic Regression ---
# calculates a linear score → passes through sigmoid → outputs probability → applies 0.5 threshold
# same math as linear regression but the output becomes a probability between 0 and 1

data_lr = {
    "hours_studied": [1, 2, 2, 3, 4, 4, 5, 6, 7, 8],
    "prev_grade":    [40, 45, 55, 60, 55, 70, 65, 75, 80, 90],
    "passed":        [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
}

df_lr = pd.DataFrame(data_lr)
X_lr  = df_lr[["hours_studied", "prev_grade"]]
y_lr  = df_lr["passed"]

X_train, X_test, y_train, y_test = train_test_split(X_lr, y_lr, test_size=0.3, random_state=42)

model_lr = LogisticRegression()
model_lr.fit(X_train, y_train)

predictions_lr = model_lr.predict(X_test)
probs_lr       = model_lr.predict_proba(X_test)[:, 1]   # probability of class 1

for pred, actual, prob in zip(predictions_lr, y_test, probs_lr):
    print(f"Predicted: {pred} | Actual: {actual} | Probability: {prob:.2f}")

print(f"Accuracy: {accuracy_score(y_test, predictions_lr):.2f}")

new_student = pd.DataFrame({"hours_studied": [5], "prev_grade": [68]})
prob_new  = model_lr.predict_proba(new_student)[0][1]
label_new = model_lr.predict(new_student)[0]
print(f"New student — pass probability: {prob_new:.2f} | Prediction: {'Pass' if label_new == 1 else 'Fail'}")


# --- Spam Email Classifier ---
# logistic regression applied to email features
# coefficients show which features are the strongest spam signals

data_spam = {
    "word_free":   [3, 0, 5, 1, 0, 4, 0, 2, 0, 6, 1, 0, 3, 0, 2],
    "word_winner": [2, 0, 3, 0, 0, 2, 0, 1, 0, 4, 0, 0, 2, 0, 1],
    "num_links":   [5, 1, 8, 2, 0, 6, 1, 3, 0, 9, 2, 1, 4, 0, 3],
    "all_caps":    [1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0],
    "num_exclaim": [4, 0, 6, 1, 0, 5, 0, 2, 0, 7, 1, 0, 3, 0, 2],
    "spam":        [1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1]
}

df_spam = pd.DataFrame(data_spam)
X_spam  = df_spam[["word_free", "word_winner", "num_links", "all_caps", "num_exclaim"]]
y_spam  = df_spam["spam"]

X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_spam, y_spam, test_size=0.2, random_state=42)

model_spam = LogisticRegression()
model_spam.fit(X_train_s, y_train_s)

preds_spam = model_spam.predict(X_test_s)
print(f"Spam classifier accuracy: {accuracy_score(y_test_s, preds_spam):.2f}")

coef_spam = pd.Series(model_spam.coef_[0], index=X_spam.columns)
print(coef_spam.sort_values(ascending=False))   # num_links = strongest spam signal

new_emails = pd.DataFrame({
    "word_free":   [4, 0],
    "word_winner": [2, 0],
    "num_links":   [7, 1],
    "all_caps":    [1, 0],
    "num_exclaim": [5, 0]
})

results = model_spam.predict(new_emails)
probs   = model_spam.predict_proba(new_emails)[:, 1]
for i, (label, prob) in enumerate(zip(results, probs)):
    print(f"Email {i+1}: {'SPAM' if label == 1 else 'NOT SPAM'} ({prob:.0%} confidence)")


# --- Decision Trees ---
# asks a series of yes/no questions learned from the data
# picks splits that create the cleanest class separation (information gain)
# max_depth controls how many questions deep the tree goes — deeper = risk of overfitting

data_dt = {
    "age":     [22, 45, 31, 60, 25, 52, 38, 29, 55, 41, 33, 48, 27, 65, 36],
    "balance": [1200, 8000, 3000, 15000, 500, 12000, 6000, 2000, 11000, 7000, 4000, 9000, 800, 18000, 5000],
    "job":     [0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0],   # 0=student, 1=employed
    "default": [1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0]
}

df_dt = pd.DataFrame(data_dt)
X_dt  = df_dt[["age", "balance", "job"]]
y_dt  = df_dt["default"]

X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X_dt, y_dt, test_size=0.2, random_state=42)

model_dt = DecisionTreeClassifier(max_depth=3, random_state=42)
model_dt.fit(X_train_d, y_train_d)

preds_dt   = model_dt.predict(X_test_d)
importance = pd.Series(model_dt.feature_importances_, index=X_dt.columns)

print(f"Decision Tree accuracy: {accuracy_score(y_test_d, preds_dt):.2f}")
print(importance.sort_values(ascending=False))   # balance = most important split

new_customers = pd.DataFrame({"age": [28, 55], "balance": [600, 14000], "job": [0, 1]})
for i, label in enumerate(model_dt.predict(new_customers)):
    print(f"Customer {i+1}: {'Default' if label == 1 else 'No Default'}")


# --- Random Forest ---
# builds 100+ trees, each on a random sample of data with random feature subsets
# takes majority vote across all trees — more stable, less overfitting than one tree
# this is called ensemble learning

model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
model_rf.fit(X_train_d, y_train_d)

preds_rf     = model_rf.predict(X_test_d)
importance_rf = pd.Series(model_rf.feature_importances_, index=X_dt.columns)

print(f"Random Forest accuracy: {accuracy_score(y_test_d, preds_rf):.2f}")
print(importance_rf.sort_values(ascending=False))

preds_rf_new = model_rf.predict(new_customers)
probs_rf_new = model_rf.predict_proba(new_customers)[:, 1]
for i, (label, prob) in enumerate(zip(preds_rf_new, probs_rf_new)):
    print(f"Customer {i+1}: {'Default' if label == 1 else 'No Default'} ({prob:.0%} confidence)")


# --- K-Nearest Neighbors ---
# no model built — memorizes training data and finds K most similar examples at prediction time
# majority vote from those K neighbors decides the label
# requires scaling — KNN measures distance, so feature ranges must be equal

scaler_knn     = StandardScaler()
X_train_scaled = scaler_knn.fit_transform(X_train_d)
X_test_scaled  = scaler_knn.transform(X_test_d)

for k in [1, 3, 5]:
    model_knn = KNeighborsClassifier(n_neighbors=k)
    model_knn.fit(X_train_scaled, y_train_d)
    preds_knn = model_knn.predict(X_test_scaled)
    print(f"KNN (k={k}) accuracy: {accuracy_score(y_test_d, preds_knn):.2f}")

model_knn5 = KNeighborsClassifier(n_neighbors=5)
model_knn5.fit(X_train_scaled, y_train_d)
new_scaled = scaler_knn.transform(new_customers)
for i, label in enumerate(model_knn5.predict(new_scaled)):
    print(f"Customer {i+1}: {'Default' if label == 1 else 'No Default'}")


# --- Titanic Survival Classifier ---
# compare all four classifiers on real data
# precision = of predicted survivors, how many actually survived
# recall    = of actual survivors, how many did the model catch
# f1-score  = harmonic mean of precision and recall — best single metric for imbalanced data

df_titanic = pd.read_csv("tested.csv")
df_titanic = df_titanic.drop(columns=["Cabin"])
df_titanic["Age"]  = df_titanic["Age"].fillna(df_titanic["Age"].median())
df_titanic["Fare"] = df_titanic["Fare"].fillna(df_titanic["Fare"].median())
df_titanic["Sex"]  = (df_titanic["Sex"] == "female").astype(int)   # female=1, male=0

X_t = df_titanic[["Pclass", "Sex", "Age", "Fare", "SibSp", "Parch"]]
y_t = df_titanic["Survived"]

print(y_t.value_counts())   # check class balance before training

X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(X_t, y_t, test_size=0.2, random_state=42)

scaler_t         = StandardScaler()
X_train_t_scaled = scaler_t.fit_transform(X_train_t)
X_test_t_scaled  = scaler_t.transform(X_test_t)

classifiers = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree":       DecisionTreeClassifier(max_depth=4, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
    "KNN":                 KNeighborsClassifier(n_neighbors=5)
}

for name, clf in classifiers.items():
    if name == "KNN":
        clf.fit(X_train_t_scaled, y_train_t)
        preds = clf.predict(X_test_t_scaled)
    else:
        clf.fit(X_train_t, y_train_t)
        preds = clf.predict(X_test_t)
    print(f"\n{name}: {accuracy_score(y_test_t, preds):.2f}")
    print(classification_report(y_test_t, preds))

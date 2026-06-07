
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report, roc_auc_score, r2_score, mean_absolute_error)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

data =pd.DataFrame( {
    "study_hours":  [1, 3, 2, 5, 4, 6, 2, 7, 3, 8, 1, 5, 4, 6, 3, 7, 2, 8, 5, 4],
    "attendance":   [50, 65, 55, 80, 70, 90, 60, 92, 68, 95, 45, 85, 72, 88, 63, 91, 58, 94, 82, 74],
    "sleep_hours":  [4, 6, 5, 7, 6, 8, 5, 8, 6, 9, 4, 7, 6, 8, 5, 8, 5, 9, 7, 6],
    "prev_score":   [40, 58, 48, 72, 65, 82, 52, 88, 60, 93, 35, 79, 67, 85, 55, 90, 50, 94, 77, 68],
    "final_score":  [42, 60, 50, 75, 67, 84, 54, 90, 62, 95, 37, 81, 69, 87, 57, 92, 52, 96, 79, 70]
})

mean_pass = data["final_score"].mean()
std_pass = data["final_score"].std()

print(mean_pass)
print(f"{std_pass:.2f}")
print(data.corr()["final_score"].sort_values(ascending= False))

passed  = data[(data["final_score"] >= 60) & (data["study_hours"] > 4 )]
studied_hard = data[data["study_hours"] > 4 ]
probability = len(passed) / len(studied_hard) * 100 
print(probability)
data["passed"] = (data["final_score"]>= 60).astype(int)

x = data[["study_hours", "attendance", "sleep_hours", "prev_score"]]
y = data["passed"]

x_train , x_test , y_train, y_test = train_test_split(x,y,test_size= 0.2, random_state= 42)

model = LinearRegression()
scaler = StandardScaler()
scaled_x = scaler.fit_transform(x_train)
scaled_x_test = scaler.transform(x_test)
model_scaled = LinearRegression()
model_scaled.fit(scaled_x, y_train)
coef = pd.Series(model_scaled.coef_, index=x.columns)
model.fit(x_train, y_train)
pred = model.predict(x_test)
r2 = r2_score(y_test, pred)
mae = mean_absolute_error(y_test, pred)
new = pd.DataFrame({
    "study_hours" :[5],
    "attendance"  :[83],
    "sleep_hours" :[7],
    "prev_score"  :[74]
})

new_pred = model.predict(new)

print(f"r2 score: {r2:.2f} mae: {mae:.2f}")
print(f"New prediction: {new_pred[0]:.0f}")
print(f"scaled coefficient: {coef}")



trained = {}
classification = {
    "logictic_regression" : (LogisticRegression()),
    "Random_forest"       : (RandomForestClassifier()),
    "KNN"                 : (KNeighborsClassifier())
}

for  name,  model in classification.items():
    if name == "KNN":
        model.fit(scaled_x, y_train)
        preds = model.predict(scaled_x_test)
    else:
        model.fit(x_train, y_train)
        preds =  model.predict(x_test)
    trained[name] = model 
   
    print(f"\n{name}: {accuracy_score(y_test, preds):.2f}")
    print(classification_report(y_test, preds))
results = {}
for name , model in trained.items():
    if name != "KNN":
     
       pred = model.predict(x_test)

       cm = confusion_matrix(y_test,pred )
       auc = roc_auc_score(y_test, pred)
       cv = cross_val_score(model, x_train,y_train,cv=5)
      
       print(f" name: {name} auc score: {auc:.2f}")
       print(f" name: {name} class report: {classification_report(y_test, pred)}")
       print(f"CV mean: {cv.mean():.4f} | std: {cv.std():.4f}")
       results[name] = {"auc": auc, "cv": cv.mean()}

    else:
        pred = model.predict(scaled_x_test)

        cm = confusion_matrix(y_test,pred )
        auc = roc_auc_score(y_test, pred)
        cv = cross_val_score(model, scaled_x,y_train,cv=5)
      
        print(f" name: {name}auc score: {auc:.2f}")
        print(f" name: {name} class report: {classification_report(y_test, pred)}")
        print(f" name{name} CV mean: {cv.mean():.4f} | std: {cv.std():.4f}")
        results[name] = {"auc": auc, "cv":cv.mean() }


params = {
    "max_depth" : [2, 3, 5, 7],
    "n_estimators": [50, 100, 200]
}
grid_search = GridSearchCV(RandomForestClassifier(random_state= 42), params, cv=5)

grid_search.fit(x_train,y_train)
best = grid_search.best_estimator_
pred_best = best.predict(x_test)
prob_best = best.predict_proba(x_test)[:, 1]

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV AUC:     {grid_search.best_score_:.4f}")
print(f"Test AUC:        {roc_auc_score(y_test, prob_best):.4f}")
print(f"Test Accuracy:   {accuracy_score(y_test, pred_best):.4f}")
print(classification_report(y_test, pred_best, target_names=["Failed", "Passed"]))

print(f"\n{'='*55}")
print(f"{'MODEL COMPARISON':^55}")
print(f"{'='*55}")
print(f"\n{'Model':<25} {'Accuracy':>9} {'AUC':>8} {'CV Mean':>9}")
print("-" * 55)

for name, scores in results.items():
    print(f"{name:<25} {scores['auc']:>8.4f} {scores['cv']:>9.4f}")

best_name = max(results, key=lambda x: results[x]["auc"])
print(f"\nBest model: {best_name}")


new_data = pd.DataFrame({
    "study_hours" : [6, 1, 4],
    "attendance" : [88, 50, 72],
    "sleep_hours" : [7, 5, 6 ],
    "prev_score" : [80, 38, 65]
})

pred = best.predict(new_data)

coef = pd.Series(best.feature_importances_, index=x.columns)
for pred_label, profile in zip(pred, ["Student A", "Student B", "Student C"]):
    print(f"{profile}: {'Pass' if pred_label == 1 else 'Fail'}")

print(coef)
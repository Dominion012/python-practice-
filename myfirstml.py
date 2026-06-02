from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error, r2_score
import pandas as pd
from sklearn.preprocessing import StandardScaler
data = {
    "study_hours":    [2, 5, 1, 8, 3, 6, 4, 7, 2, 9, 1, 5, 3, 7, 4],
    "attendance":     [60, 90, 45, 95, 70, 85, 75, 92, 55, 98, 40, 88, 65, 91, 78],
    "prev_score":     [55, 78, 40, 92, 62, 80, 70, 88, 50, 95, 35, 82, 60, 89, 72],
    "sleep_hours":    [5, 7, 4, 8, 6, 7, 6, 8, 5, 9, 4, 7, 6, 8, 7],
    "final_score":    [52, 80, 38, 94, 65, 83, 72, 91, 48, 97, 33, 85, 62, 92, 75]
}

df = pd.DataFrame(data)
x= df[["study_hours", "attendance", "prev_score", "sleep_hours"]]
y = df['final_score']

x_train, x_test , y_train, y_test = train_test_split(x,y, test_size=0.2 , random_state=42)

model = LinearRegression()
model.fit(x_train, y_train)
prediction = model.predict(x_test)

for pred, actual in zip(prediction, y_test):
    print(f"predicted_score {pred:.0f}  | actual score {actual:.0f}")

err = mean_absolute_error(prediction, y_test)
print(f"mean absolute error: {err:.2f}")


r2 = r2_score(y_test, prediction)
coef = pd.Series(model.coef_, index=x.columns)
scaled = StandardScaler()
x_scaled = scaled.fit_transform(x)
model_unscaled = LinearRegression()
model_unscaled.fit(x,y)

model_scaled = LinearRegression()
model_scaled.fit(x_scaled,y)
#print(coef)
print(f" r2 squared is {r2:.2f}")

print("Unscaled coefficients:")
print(pd.Series(model_unscaled.coef_, index=x.columns))

print("\nScaled coefficients:")
print(pd.Series(model_scaled.coef_, index=x.columns))


new_student = pd.DataFrame({
    "study_hours": [6],
    "attendance": [88],
    "prev_score": [75],
    "sleep_hours": [7]
})
print(f"Predicted final score: {model.predict(new_student)[0]:.0f}")

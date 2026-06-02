import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

# =============================================
# Day 17 - Linear Regression Deep Dive
# =============================================
# --- Concept 1: How Linear Regression Works ---
# Hours studied vs exam score — fitting a line manually

# --- Concept 2: y = mx + b ---



# --- Concept 3: Simple Linear Regression ---
# Predict house price from square footage only
# --- Concept 4: Multiple Linear Regression ---
# --- Concept 5: R² Score ---
# --- Concept 4: Multiple Linear Regression ---

data = {
    "sqft":      [800, 1000, 1200, 1500, 1800, 2000, 2500, 3000, 1100, 1700],
    "bedrooms":  [2, 2, 3, 3, 3, 4, 4, 5, 2, 3],
    "age":       [30, 20, 15, 5, 10, 2, 1, 8, 25, 12],
    "price":     [160000, 200000, 245000, 310000, 360000, 405000, 495000, 590000, 215000, 340000]
}

df = pd.DataFrame(data)
X = df[["sqft", "bedrooms", "age"]]
y = df["price"]

model_multi = LinearRegression()
model_multi.fit(X, y)

coef = pd.Series(model_multi.coef_, index=X.columns)
print(coef)
#print(f"Intercept: ${model_multi.intercept_:,.0f}")

# Predict: 1,600 sqft, 3 bedrooms, 7 years old
new_house = pd.DataFrame({"sqft": [1600], "bedrooms": [3], "age": [7]})
pred = model_multi.predict(new_house)[0]
#print(f"Predicted price: ${pred:,.0f}")


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_r2 = LinearRegression()
model_r2.fit(X_train, y_train)
preds = model_r2.predict(X_test)

mae = mean_absolute_error(y_test, preds)
r2  = r2_score(y_test, preds)

#print(f"MAE: ${mae:,.0f}")
#print(f"R²:  {r2:.4f}")

# Compare: what if we just predicted the mean every time?
mean_pred = np.full(len(y_test), y_train.mean())
r2_baseline = r2_score(y_test, mean_pred)
#print(f"R² if we just guessed the mean: {r2_baseline:.4f}")

# --- Concept 6: Residuals ---

model_res = LinearRegression()
model_res.fit(X, y)
all_preds = model_res.predict(X)

residuals = y.values - all_preds

#for i, (actual, pred, resid) in enumerate(zip(y.values, all_preds, residuals)):
    #print(f"House {i+1}: actual=${actual:,}  predicted=${pred:,.0f}  residual=${resid:,.0f}")
# --- Concept 7: Assumptions ---

# Violation 1: non-linear relationship
sqft2  = np.array([500, 800, 1000, 1200, 1500, 2000, 2500, 3000])
price2 = np.array([100000, 160000, 210000, 280000, 390000, 600000, 900000, 1300000])

model_nl = LinearRegression()
model_nl.fit(sqft2.reshape(-1, 1), price2)
r2_nl = r2_score(price2, model_nl.predict(sqft2.reshape(-1, 1)))
#print(f"R² on non-linear data: {r2_nl:.4f}")

# Violation 2: multicollinearity
data2 = {
    "sqft":       [800, 1000, 1200, 1500, 1800, 2000],
    "sqft_copy":  [800, 1000, 1200, 1500, 1800, 2000],  # identical feature
    "price":      [160000, 200000, 245000, 310000, 360000, 405000]
}
df2 = pd.DataFrame(data2)
X2  = df2[["sqft", "sqft_copy"]]
y2  = df2["price"]

model_mc = LinearRegression()
model_mc.fit(X2, y2)
coef2 = pd.Series(model_mc.coef_, index=X2.columns)
#print(coef2)  # coefficients will look unstable/split
# --- Concept 8: Feature Scaling ---

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model_unscaled = LinearRegression()
model_unscaled.fit(X, y)

model_scaled = LinearRegression()
model_scaled.fit(X_scaled, y)

print("Unscaled coefficients:")
print(pd.Series(model_unscaled.coef_, index=X.columns))

print("\nScaled coefficients:")
print(pd.Series(model_scaled.coef_, index=X.columns))

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

# =============================================
# Day 17 - Linear Regression Deep Dive
# =============================================


# --- How Linear Regression Works ---
# finds the line that minimizes total squared error across all data points

hours  = np.array([1, 2, 3, 4, 5, 6, 7, 8])
scores = np.array([40, 50, 55, 60, 65, 70, 78, 85])

def total_squared_error(hours, scores, slope, intercept):
    predictions = slope * hours + intercept
    errors = scores - predictions
    return np.sum(errors ** 2)

bad  = total_squared_error(hours, scores, slope=5, intercept=30)
good = total_squared_error(hours, scores, slope=6, intercept=35)
new  = total_squared_error(hours, scores, slope=7, intercept=32)

print(f"Bad line error:  {bad:.1f}")
print(f"Good line error: {good:.1f}")    # smallest = best fit
print(f"New line error:  {new:.1f}")


# --- y = mx + b ---
# m = coefficient (how much y changes per 1 unit of x)
# b = intercept (base value when x = 0)

experience = np.array([0, 1, 2, 3, 5, 7, 10, 15])
salary     = np.array([30000, 35000, 40000, 45000, 55000, 65000, 80000, 105000])

model_eq = LinearRegression()
model_eq.fit(experience.reshape(-1, 1), salary)

manual_pred  = model_eq.coef_[0] * 8 + model_eq.intercept_
sklearn_pred = model_eq.predict([[8]])[0]

print(f"Intercept (base salary):       ${model_eq.intercept_:,.0f}")
print(f"Coefficient (salary per year): ${model_eq.coef_[0]:,.0f}")
print(f"Manual prediction (8 years):   ${manual_pred:,.0f}")
print(f"Sklearn prediction (8 years):  ${sklearn_pred:,.0f}")   # identical


# --- Simple Linear Regression ---
# one feature predicting one output

sqft  = np.array([600, 800, 1000, 1200, 1500, 1800, 2000, 2500, 3000])
price = np.array([120000, 160000, 200000, 240000, 295000, 355000, 395000, 490000, 585000])

model_simple = LinearRegression()
model_simple.fit(sqft.reshape(-1, 1), price)

mae_simple = mean_absolute_error(price, model_simple.predict(sqft.reshape(-1, 1)))
pred_1750  = model_simple.predict([[1750]])[0]

print(f"Base price (intercept):         ${model_simple.intercept_:,.0f}")
print(f"Price per sqft:                 ${model_simple.coef_[0]:,.2f}")
print(f"MAE:                            ${mae_simple:,.0f}")
print(f"Predicted price for 1750 sqft:  ${pred_1750:,.0f}")


# --- Multiple Linear Regression ---
# each feature gets its own coefficient, learned independently

data = {
    "sqft":     [800, 1000, 1200, 1500, 1800, 2000, 2500, 3000, 1100, 1700],
    "bedrooms": [2, 2, 3, 3, 3, 4, 4, 5, 2, 3],
    "age":      [30, 20, 15, 5, 10, 2, 1, 8, 25, 12],
    "price":    [160000, 200000, 245000, 310000, 360000, 405000, 495000, 590000, 215000, 340000]
}

df = pd.DataFrame(data)
X  = df[["sqft", "bedrooms", "age"]]
y  = df["price"]

model_multi = LinearRegression()
model_multi.fit(X, y)

new_house = pd.DataFrame({"sqft": [1600], "bedrooms": [3], "age": [7]})

print(pd.Series(model_multi.coef_, index=X.columns))
print(f"Intercept: ${model_multi.intercept_:,.0f}")
print(f"Predicted price (1600sqft, 3bed, 7yr): ${model_multi.predict(new_house)[0]:,.0f}")


# --- R² Score ---
# how much of the variation in y the model explains
# 1.0 = perfect, 0.0 = no better than guessing the mean

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_r2 = LinearRegression()
model_r2.fit(X_train, y_train)
preds = model_r2.predict(X_test)

mae         = mean_absolute_error(y_test, preds)
r2          = r2_score(y_test, preds)
mean_pred   = np.full(len(y_test), y_train.mean())
r2_baseline = r2_score(y_test, mean_pred)

print(f"MAE: ${mae:,.0f}")
print(f"R²:  {r2:.4f}")
print(f"R² if always predicting mean: {r2_baseline:.4f}")


# --- Residuals ---
# residual = actual - predicted
# mean residual ≈ 0 means model is unbiased (not systematically over/under predicting)

model_res = LinearRegression()
model_res.fit(X, y)
all_preds = model_res.predict(X)
residuals = y.values - all_preds

for i, (actual, pred, resid) in enumerate(zip(y.values, all_preds, residuals)):
    print(f"House {i+1}: actual=${actual:,}  predicted=${pred:,.0f}  residual=${resid:,.0f}")

print(f"Mean residual:    ${np.mean(residuals):,.2f}")    # close to 0 = unbiased
print(f"Largest error:    ${np.max(np.abs(residuals)):,.0f}")
print(f"Std of residuals: ${np.std(residuals):,.0f}")


# --- Assumptions of Linear Regression ---

# violation 1: non-linear relationship — R² looks ok but line misses the curve
sqft2  = np.array([500, 800, 1000, 1200, 1500, 2000, 2500, 3000])
price2 = np.array([100000, 160000, 210000, 280000, 390000, 600000, 900000, 1300000])

model_nl = LinearRegression()
model_nl.fit(sqft2.reshape(-1, 1), price2)
r2_nl = r2_score(price2, model_nl.predict(sqft2.reshape(-1, 1)))
print(f"R² on non-linear data: {r2_nl:.4f}")

# violation 2: multicollinearity — identical features split the coefficient unstably
data2 = {
    "sqft":      [800, 1000, 1200, 1500, 1800, 2000],
    "sqft_copy": [800, 1000, 1200, 1500, 1800, 2000],
    "price":     [160000, 200000, 245000, 310000, 360000, 405000]
}
df2 = pd.DataFrame(data2)
X2  = df2[["sqft", "sqft_copy"]]
y2  = df2["price"]

model_mc = LinearRegression()
model_mc.fit(X2, y2)
print(pd.Series(model_mc.coef_, index=X2.columns))   # weight split between identical features


# --- Feature Scaling ---
# standardize so coefficients can be compared fairly across features
# raw coefficients mislead when features have different ranges

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

model_unscaled = LinearRegression()
model_unscaled.fit(X, y)

model_scaled = LinearRegression()
model_scaled.fit(X_scaled, y)

print("Unscaled coefficients:")
print(pd.Series(model_unscaled.coef_, index=X.columns))

print("\nScaled coefficients (comparable):")
print(pd.Series(model_scaled.coef_, index=X.columns))
# sqft has the highest scaled coef — most impactful feature despite lower raw coef

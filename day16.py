from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import pandas as pd

# =============================================
# Day 16 - Introduction to Machine Learning
# =============================================


# --- Features and Labels ---

data = {
    "bedrooms":  [2, 3, 3, 4, 2, 5, 3, 4],
    "bathrooms": [1, 2, 1, 3, 1, 3, 2, 2],
    "sqft":      [900, 1400, 1100, 2000, 850, 2500, 1300, 1800],
    "age":       [20, 5, 15, 2, 30, 1, 8, 12],
    "price":     [180000, 320000, 250000, 480000, 160000, 600000, 300000, 420000]
}

df = pd.DataFrame(data)

X = df[["bedrooms", "bathrooms", "sqft", "age"]]   # features — inputs
y = df["price"]                                     # label — what we predict

print(X.shape)   # (8, 4) — 8 houses, 4 features


# --- Split Data (80% train, 20% test) ---

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train.shape)   # 6 houses for training
print(X_test.shape)    # 2 houses for testing


# --- Train the Model ---

model = LinearRegression()
model.fit(X_train, y_train)   # model learns from training data


# --- Make Predictions ---

predictions = model.predict(X_test)

for pred, actual in zip(predictions, y_test):
    print(f"Predicted: ${pred:,.0f}  |  Actual: ${actual:,.0f}")


# --- Evaluate the Model ---

mae = mean_absolute_error(y_test, predictions)
print(f"Mean Absolute Error: ${mae:,.0f}")   # on average how many dollars off


# --- What Did the Model Learn ---

# coefficients — how much each feature affects the price
coef = pd.Series(model.coef_, index=X.columns)
print(coef)

# positive coefficient = increases price
# negative coefficient = decreases price (age makes house cheaper)
print(f"Intercept: ${model.intercept_:,.0f}")   # base price before any features


# --- Predict on New Unseen Data ---

new_house = pd.DataFrame({
    "bedrooms":  [4],
    "bathrooms": [3],
    "sqft":      [2200],
    "age":       [0]
})

predicted_price = model.predict(new_house)
print(f"Predicted price for new house: ${predicted_price[0]:,.0f}")

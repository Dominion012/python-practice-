import numpy as np
from scipy import stats
import pandas as pd

# =============================================
# Day 13 - Statistics for Machine Learning
# =============================================


# --- Mean, Median and Mode ---

salaries = np.array([35000, 42000, 38000, 45000, 40000, 39000, 42000, 500000])

print(np.mean(salaries))            # pulled up by 500000 outlier
print(np.median(salaries))          # not affected by outlier — best representation
print(stats.mode(salaries).mode)    # most frequent value

# use mean when no outliers, median when outliers exist, mode for categories


# --- Variance and Standard Deviation ---

model_a = np.array([88, 90, 87, 91, 89, 90, 88])   # consistent
model_b = np.array([60, 95, 72, 99, 55, 91, 83])   # inconsistent

print(np.mean(model_a))   # similar means
print(np.mean(model_b))

print(np.std(model_a))    # low std — reliable model
print(np.std(model_b))    # high std — unpredictable model

print(np.var(model_a))    # variance = std squared
print(np.var(model_b))


# --- Normal Distribution ---

# generate IQ scores — mean 100, std 15
data = np.random.normal(loc=100, scale=15, size=1000)

print(np.mean(data))    # close to 100
print(np.std(data))     # close to 15

# 68% rule — values within 1 std of the mean
within_1std = data[(data >= 85) & (data <= 115)]
print(len(within_1std) / len(data) * 100)   # close to 68%


# --- Correlation ---

df = pd.DataFrame({
    "age":         [22, 25, 30, 35, 40, 45, 50],
    "salary":      [30000, 35000, 50000, 65000, 80000, 90000, 95000],
    "coffee_cups": [3, 2, 4, 3, 2, 1, 2],
    "experience":  [1, 3, 7, 12, 18, 22, 27]
})

print(df.corr())   # experience has highest correlation with salary
                   # coffee cups has near zero — not useful for prediction


# --- Percentiles and Quartiles ---

house_prices = np.array([150000, 180000, 200000, 220000, 250000,
                          270000, 300000, 350000, 400000, 2000000])

q1 = np.percentile(house_prices, 25)
q2 = np.percentile(house_prices, 50)   # median
q3 = np.percentile(house_prices, 75)
iqr = q3 - q1

lower = q1 - 1.5 * iqr   # outlier boundary
upper = q3 + 1.5 * iqr

print(q1, q2, q3, iqr)

# find outliers
outliers = []
for house in house_prices:
    if house < lower or house > upper:
        outliers.append(house)
print(outliers)   # [2000000]


# --- Skewness ---

incomes = np.array([25000, 28000, 30000, 32000, 35000,
                    38000, 40000, 45000, 50000, 500000])

print(stats.skew(incomes))              # positive — right skewed
print(np.mean(incomes))                 # mean > median because of outlier
print(np.median(incomes))

# log transformation reduces skewness
log_incomes = np.log(incomes)
print(stats.skew(log_incomes))          # closer to 0 after transformation

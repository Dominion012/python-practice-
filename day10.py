import pandas as pd
import numpy as np

# =============================================
# Day 10 - Titanic Data Project
# =============================================


# --- Loading and Exploring the Dataset ---

df = pd.read_csv("tested.csv")

print(df.shape)        # (418, 12) — 418 rows, 12 columns
print(df.columns)      # list of column names
print(df.head())       # first 5 rows
print(df.info())       # column types and missing value counts
print(df.describe())   # stats: mean, min, max, std for number columns


# --- Finding and Handling Missing Values ---

print(df.isnull().sum())              # count missing values per column

df = df.drop(columns=["Cabin"])       # too many missing — drop the whole column
df["Age"]  = df["Age"].fillna(df["Age"].median())    # fill with median age
df["Fare"] = df["Fare"].fillna(df["Fare"].median())  # fill with median fare

print(df.isnull().sum())              # confirm all zeros


# --- Basic Analysis ---

# survival rate — mean of 0s and 1s gives percentage
print(df["Survived"].mean())          # 0.36 = 36% survived
print(df["Survived"].mean() * 100)    # as a percentage

# count survivors vs non-survivors
print(df["Survived"].value_counts())  # 0 = died, 1 = survived

# average age of all passengers
print(df["Age"].mean())

# male vs female breakdown
print(df["Sex"].value_counts())

# passenger class breakdown
print(df["Pclass"].value_counts())

# embarkation port breakdown
print(df["Embarked"].value_counts())


# --- Grouping and Aggregating ---

# survival rate by gender
print(df.groupby("Sex")["Survived"].mean() * 100)

# survival rate by passenger class
print(df.groupby("Pclass")["Survived"].mean() * 100)

# average age per passenger class
print(df.groupby("Pclass")["Age"].mean())

# average fare per passenger class
print(df.groupby("Pclass")["Fare"].mean())

# average fare by gender
print(df.groupby("Sex")["Fare"].mean())

# average age by gender
print(df.groupby("Sex")["Age"].mean())

# survival rate grouped by class AND gender combined
print(df.groupby(["Pclass", "Sex"])["Survived"].mean() * 100)


# --- Conclusions ---
# - 1st class paid 8x more than 3rd class (94 vs 12 average fare)
# - 1st class passengers were older on average than 3rd class
# - More males (266) than females (152) on board
# - Most passengers boarded from Southampton (port S)
# - Average passenger age was around 30

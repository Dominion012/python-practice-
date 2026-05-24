import numpy as np
import pandas as pd

# =============================================
# Week 2 Exercise - NBA Player Stats Analyzer
# Covers: NumPy, Pandas, Data Cleaning, Groupby, Analysis
# =============================================

df = pd.read_csv("nba_stats.csv")


# --- Data Cleaning (fill missing values first) ---

df["points"]       = df["points"].fillna(df["points"].mean())
df["rebounds"]     = df["rebounds"].fillna(df["rebounds"].mean())
df["games_played"] = df["games_played"].fillna(df["games_played"].median())
df = df.rename(columns={"games_played": "games"})


# --- NumPy Analysis (Days 6 & 7) ---

# convert points column to NumPy array
numbers = df["points"].to_numpy()

print(np.mean(numbers))           # average points
print(np.max(numbers))            # highest points
print(np.min(numbers))            # lowest points
print(np.std(numbers))            # spread of points

# boolean indexing — players scoring above 28
print(numbers[numbers > 28])


# --- Pandas Basics (Day 8) ---

print(df.shape)                                      # rows and columns
print(df.info())                                     # column types and missing values
print(df.describe())                                 # stats summary

print(df[["name", "points"]])                        # select two columns
print(df.iloc[0:5])                                  # first 5 rows by position
print(df.loc[3:7, ["name", "team", "points"]])       # rows 3-7, specific columns


# --- Data Cleaning (Day 9) ---

print(df.isnull().sum())                             # missing values per column
print(df.dtypes)                                     # check all data types


# --- Groupby & Aggregation (Days 9 & 10) ---

print(df.groupby("position")["points"].mean())       # average points per position
print(df.groupby("team")["salary"].mean())           # average salary per team
print(df.groupby("position")["games"].sum())         # total games per position
print(df.groupby("position")["salary"].agg(["min", "max"]))             # min and max salary per position
print(df.groupby("position")[["points", "rebounds"]].agg(["mean"]))     # avg points and rebounds per position


# --- Analysis & Conclusions (Day 10) ---

# which position scores the most
print(df.groupby("position")["points"].mean().idxmax())

# which team has the highest average salary
print(df.groupby("team")["salary"].mean().idxmax())

# players averaging above 28 points
print(df[df["points"] > 28][["name", "points"]])

# top 5 scorers
print(df.sort_values("points", ascending=False).head(5)[["name", "points"]])

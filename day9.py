import pandas as pd
import numpy as np

# =============================================
# Day 9 - Pandas Data Cleaning
# =============================================


# --- Missing Values ---

df = pd.DataFrame({
    "name":   ["Alice", "Bob", None, "David", "Eve"],
    "age":    [25, None, 22, 35, None],
    "salary": [50000, 60000, None, 80000, 55000],
    "city":   ["London", "Paris", "Tokyo", None, "Berlin"]
})

print(df.isnull())           # True/False for every cell
print(df.isnull().sum())     # count of missing values per column

print(df.dropna())           # drop rows with any missing value
print(df.dropna(how="all"))  # drop only if ALL values in a row are missing

# fill missing values
df["age"]    = df["age"].fillna(df["age"].mean())       # fill with average
df["salary"] = df["salary"].fillna(df["salary"].mean()) # fill with average
df["name"]   = df["name"].fillna("Unknown")             # fill with fixed value
df["city"]   = df["city"].fillna("Unknown")
print(df)


# --- Removing Duplicates ---

df = pd.DataFrame({
    "product": ["Apple", "Banana", "Apple", "Orange", "Banana", "Apple"],
    "price":   [1.2, 0.5, 1.2, 0.8, 0.5, 1.5],
    "stock":   [100, 200, 100, 150, 200, 80]
})

print(df.duplicated().sum())            # count duplicate rows
print(df.drop_duplicates())             # remove all duplicate rows
print(df.drop_duplicates(subset=["product"]))  # remove by specific column


# --- Renaming Columns ---

df = pd.DataFrame({
    "Student Name": ["Alice", "Bob", "Clara"],
    "AGE":          [20, 22, 21],
    "GRADE":        [88, 72, 95],
    "Home City":    ["London", "Paris", "Tokyo"]
})

df = df.rename(columns={"Student Name": "name", "Home City": "city"})
df.columns = df.columns.str.lower()        # make all column names lowercase
df.columns = df.columns.str.replace(" ", "_")  # replace spaces with underscores
print(df)


# --- Changing Data Types ---

df = pd.DataFrame({
    "product": ["Apple", "Banana", "Orange"],
    "price":   ["1.20", "0.50", "0.80"],   # stored as text
    "stock":   ["100", "200", "150"],       # stored as text
    "on_sale": [1, 0, 1]                    # stored as int
})

print(df.dtypes)                            # check types before

df["price"]   = df["price"].astype(float)
df["stock"]   = df["stock"].astype(int)
df["on_sale"] = df["on_sale"].astype(bool)

print(df.dtypes)                            # check types after
print(df)


# --- Sorting Data ---

df = pd.DataFrame({
    "name":       ["Alice", "Bob", "Clara", "David", "Eve", "Frank"],
    "department": ["HR", "IT", "IT", "HR", "Finance", "Finance"],
    "salary":     [50000, 80000, 75000, 55000, 90000, 70000],
    "years":      [3, 7, 5, 2, 9, 4]
})

print(df.sort_values("salary", ascending=False))                          # highest to lowest
print(df.sort_values(["department", "salary"], ascending=[True, False]))  # multiple columns


# --- Filtering Data ---

print(df[df["salary"] > 70000])                                           # salary above 70000

# filter and reset index so it starts from 0 again
it = df[(df["department"] == "IT") & (df["years"] > 4)].reset_index(drop=True)
print(it)


# --- Groupby and Aggregation ---

print(df.groupby("department")["salary"].mean())   # average salary per department
print(df.groupby("department")["years"].sum())     # total years per department
print(df.groupby("department")["name"].count())    # number of employees per department
print(df.groupby("department")["salary"].max())    # highest salary per department

# multiple aggregations at once
print(df.groupby("department")["salary"].agg(["mean", "min", "max", "count"]))

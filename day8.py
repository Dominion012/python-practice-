import pandas as pd

# =============================================
# Day 8 - Pandas Deep Dive
# =============================================


# --- Creating a DataFrame from scratch ---

data = {
    "name":    ["Alice", "Bob", "Clara", "David"],
    "age":     [25, 30, 22, 35],
    "city":    ["London", "New York", "Paris", "Tokyo"],
    "score":   [88, 72, 95, 61]
}

df = pd.DataFrame(data)
print(df)


# --- Reading a CSV file ---

df = pd.read_csv("students.csv")
print(df)


# --- Exploring Data ---

print(df.head())        # first 5 rows
print(df.head(3))       # first 3 rows
print(df.tail())        # last 5 rows
print(df.tail(2))       # last 2 rows
print(df.info())        # column names, data types, missing values
print(df.describe())    # stats: count, mean, min, max, std for number columns
print(df.shape)         # (rows, columns)
print(df.columns)       # list of column names
print(len(df))          # number of rows


# --- Selecting Columns ---

print(df["name"])                    # single column — returns a Series
print(df[["name", "grade"]])         # multiple columns — returns a DataFrame


# --- Selecting Rows ---

print(df.iloc[0])        # first row by position
print(df.iloc[0:4])      # first 4 rows by position
print(df.loc[0])         # first row by label
print(df.loc[0:2])       # rows 0 to 2 inclusive (loc includes stop)

# select specific rows AND columns
print(df.iloc[0:3, 0:2])                    # first 3 rows, first 2 columns
print(df.loc[2:5, ["name", "grade"]])       # rows 2-5, name and grade only


# --- Filtering Data ---

# single condition
print(df[df["grade"] > 75])                  # students with grade above 75
print(df[df["passed"] == True])              # students who passed
print(df[df["passed"] == False])             # students who failed

# multiple conditions — wrap each condition in ()
print(df[(df["age"] <= 21) & (df["grade"] > 70)])    # age 21 or under AND grade above 70
print(df[(df["grade"] < 50) | (df["grade"] > 90)])   # grade below 50 OR above 90

# filter and select specific columns at the same time
print(df[df["passed"] == False][["name", "grade"]])  # name and grade of failed students

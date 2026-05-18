import numpy as np

# =============================================
# Day 6 - NumPy Deep Dive
# =============================================


# --- Creating Arrays ---

a = np.array([1, 2, 3, 4, 5])         # array from a list
b = np.zeros(5)                         # array of zeros
c = np.ones(5)                          # array of ones
d = np.arange(0, 10, 2)                # range with step: [0 2 4 6 8]
e = np.arange(1, 11)                    # numbers 1 to 10

print(a)
print(b)
print(c)
print(d)
print(e)


# --- Array Shapes and Dimensions ---

f = np.arange(1, 13)                   # 1D array: [1 2 3 ... 12]
g = f.reshape(3, 4)                    # reshape into 3 rows, 4 columns

print(g)
print(g.shape)                         # (3, 4)
print(g.ndim)                          # 2


# --- Basic Math Operations ---

scores = np.array([55, 72, 88, 91, 64])

print(scores + 5)                      # add 5 to every element
print(scores * 2)                      # multiply every element by 2
print(scores ** 2)                     # square every element

# math between two arrays
x = np.array([1, 2, 3, 4, 5])
y = np.array([10, 20, 30, 40, 50])
print(x + y)                           # [11 22 33 44 55]
print(y / x)                           # [10. 10. 10. 10. 10.]

# stats
print(np.sum(scores))                  # total
print(np.mean(scores))                 # average
print(np.max(scores))                  # highest
print(np.min(scores))                  # lowest


# --- Indexing and Slicing ---

arr = np.array([10, 20, 30, 40, 50])

print(arr[0])                          # 10 — first item
print(arr[-1])                         # 50 — last item
print(arr[1:4])                        # [20 30 40] — slice

# 2D indexing
grid = np.array([[10, 20, 30],
                 [40, 50, 60],
                 [70, 80, 90]])

print(grid[1, 1])                      # 50 — row 1, column 1
print(grid[2, :])                      # [70 80 90] — entire last row
print(grid[:, 0])                      # [10 40 70] — entire first column

# boolean indexing — filter by condition
print(grid[grid > 45])                 # [50 60 70 80 90]

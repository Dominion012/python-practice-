import numpy as np

# =============================================
# Day 7 - NumPy Operations, Slicing & Indexing
# =============================================


# --- Math Operations on Arrays ---

a = np.array([10, 20, 30, 40, 50])
b = np.array([2, 4, 5, 8, 10])

print(a + b)     # [12 24 35 48 60]
print(a - b)     # [ 8 16 25 32 40]
print(a * b)     # [ 20  80 150 320 500]
print(a / b)     # [ 5.  5.  6.  5.  5.]
print(a % b)     # remainder of each division
print(a // b)    # floor division (no decimals)

# broadcasting — single number applied to every element
print(a + 100)   # [110 120 130 140 150]
print(a / 10)    # [ 1.  2.  3.  4.  5.]
print(a ** 2)    # [ 100  400  900 1600 2500]


# --- NumPy Built-in Functions ---

scores = np.array([78, 92, 65, 88, 73, 95, 61, 84, 77, 90])

print(np.sum(scores))     # total of all values
print(np.mean(scores))    # average
print(np.max(scores))     # highest value
print(np.min(scores))     # lowest value
print(np.std(scores))     # how spread out the values are
print(np.median(scores))  # middle value when sorted
print(np.sort(scores))    # sorted copy of the array

# argmax/argmin — returns the index (position) of max/min value
print(np.argmax(scores))  # index of highest score
print(np.argmin(scores))  # index of lowest score


# --- Array Slicing ---

temps = np.array([22, 19, 25, 30, 28, 17, 21, 26, 29, 24])

print(temps[:4])     # first 4 items
print(temps[-3:])    # last 3 items
print(temps[::2])    # every other item (step of 2)
print(temps[::-1])   # entire array reversed
print(temps[1:8:2])  # start at 1, stop at 8, step 2


# --- 2D Array Indexing ---

sales = np.array([[200, 150, 300, 250],
                  [180, 220, 170, 290],
                  [310, 140, 260, 200]])

# single item — [row, column]
print(sales[1, 2])      # 170 — row 1, column 2

# entire row
print(sales[2, :])      # [310 140 260 200] — all of row 2

# entire column
print(sales[:, 0])      # [200 180 310] — all of column 0

# slice a section — rows 0-1, columns 2-3
print(sales[0:2, 2:4])  # [[300 250]
                         #  [170 290]]


# --- Boolean Indexing ---

temperatures = np.array([35, 22, 41, 18, 29, 37, 15, 44, 26, 33])

print(temperatures[temperatures > 30])                            # values above 30
print(temperatures[temperatures < 20])                            # values below 20
print(temperatures[(temperatures >= 25) & (temperatures <= 40)]) # between 25 and 40
print(temperatures[(temperatures < 15) | (temperatures > 40)])   # below 15 or above 40
print(len(temperatures[temperatures > 30]))                       # count of values above 30


# --- Reshaping Arrays ---

num = np.arange(1, 21)        # [1 2 3 ... 20]

reshaped = num.reshape(4, 5)  # 4 rows, 5 columns
print(reshaped)

flat = reshaped.flatten()     # back to 1D
print(flat)

# -1 lets NumPy calculate the missing dimension automatically
print(num.reshape(5, -1))     # 5 rows, NumPy figures out 4 columns
print(num.reshape(-1, 4))     # 4 columns, NumPy figures out 5 rows

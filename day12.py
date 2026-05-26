import numpy as np

# =============================================
# Day 12 - Matrices & Matrix Multiplication
# =============================================


# --- Creating Matrices ---

A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

print(A)
print(A.shape)          # (3, 3)
print(A.ndim)           # 2

print(np.zeros((3, 3))) # 3x3 matrix of zeros
print(np.ones((2, 4)))  # 2x4 matrix of ones
print(np.eye(3))        # 3x3 identity matrix — multiplying by it changes nothing


# --- Matrix Addition and Subtraction ---

weights = np.array([[0.5, 0.3],
                    [0.8, 0.1]])

bias = np.array([[0.1, 0.2],
                 [0.1, 0.2]])

print(weights + bias)   # add bias to weights
print(weights - bias)   # subtract bias from weights


# --- Scalar Multiplication ---

A = np.array([[2, 4],
              [6, 8]])

print(A * 0.1)   # scale down
print(A * 3)     # scale up
print(A * -1)    # flip all values to negative


# --- Matrix Multiplication ---

# inner dimensions must match — result shape is outer dimensions
# A(2,3) @ B(3,2) → result (2,2)
A = np.array([[1, 2, 3],
              [4, 5, 6]])        # shape (2, 3)

B = np.array([[7,  8],
              [9,  10],
              [11, 12]])         # shape (3, 2)

print(A @ B)            # (2, 2) result
print(A @ B.shape)


# --- Element-wise vs Matrix Multiplication ---

A = np.array([[2, 3],
              [1, 4]])

B = np.array([[1, 0],
              [0, 1]])           # identity matrix

print(A * B)    # element-wise — multiplies matching positions
print(A @ B)    # matrix multiplication — gives back A unchanged (B is identity)


# --- Transpose ---

A = np.array([[1, 2, 3, 4],
              [5, 6, 7, 8]])    # shape (2, 4)

print(A.T)          # flips rows and columns
print(A.T.shape)    # (4, 2)
print(A @ A.T)      # (2,4) @ (4,2) → (2, 2)


# --- Neural Network Forward Pass ---

# 4 patients, 3 health measurements each
patients = np.array([[120, 80, 25],
                     [140, 90, 30],
                     [110, 70, 22],
                     [130, 85, 28]])   # shape (4, 3)

# weights — 3 inputs → 2 outputs
weights = np.array([[0.2, 0.5],
                    [0.1, 0.3],
                    [0.4, 0.6]])       # shape (3, 2)

bias = np.array([0.1, 0.2])

# one forward pass — predict for all 4 patients at once
# this is exactly what every neural network layer does
output = patients @ weights + bias    # shape (4, 2)
print(output)
print(output.shape)

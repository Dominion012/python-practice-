import numpy as np

# =============================================
# Day 11 - Vectors & Linear Algebra for AI
# =============================================


# --- Creating Vectors ---

# vectors represent real world data as numbers
house1 = np.array([3, 2, 1500, 300000])    # bedrooms, bathrooms, sqft, price
house2 = np.array([4, 3, 2000, 450000])

print(house1)
print(house1.shape)                         # (4,) — 4 elements, 1 dimension


# --- Vector Addition and Subtraction ---

# word vectors — famous AI example: king - man + woman = queen
king  = np.array([0.9, 0.1, 0.8, 0.3])
queen = np.array([0.9, 0.9, 0.8, 0.3])
man   = np.array([0.1, 0.1, 0.5, 0.2])
woman = np.array([0.1, 0.9, 0.5, 0.2])

result = king - man + woman
print(result)   # should match queen
print(queen)


# --- Scalar Multiplication ---

# one step of neural network training
weights       = np.array([0.8, 0.3, 0.5, 0.9])
gradient      = np.array([0.2, 0.1, 0.4, 0.3])
learning_rate = 0.01

correction   = learning_rate * gradient    # how much to adjust each weight
new_weights  = weights - correction        # updated weights after one training step

print(correction)
print(new_weights)


# --- Dot Product ---

# recommendation system — match user preferences to movies
user_preferences = np.array([0.9, 0.1, 0.8, 0.3])
movie_action     = np.array([0.9, 0.1, 0.7, 0.2])
movie_romance    = np.array([0.1, 0.9, 0.2, 0.8])

action_score  = np.dot(user_preferences, movie_action)
romance_score = np.dot(user_preferences, movie_romance)

print(action_score)    # higher = more similar to user preferences
print(romance_score)   # lower = less similar


# --- Vector Magnitude ---

v1 = np.array([1.0, 2.0, 3.0])
v2 = np.array([4.0, 0.0, 3.0])

print(np.linalg.norm(v1))   # length of v1
print(np.linalg.norm(v2))   # length of v2 — longer


# --- Unit Vectors and Normalization ---

# two users with same preferences but different rating scales
user1 = np.array([5.0, 5.0, 5.0, 5.0])   # rates everything high
user2 = np.array([1.0, 1.0, 1.0, 1.0])   # rates everything low

unit_1 = user1 / np.linalg.norm(user1)    # normalize
unit_2 = user2 / np.linalg.norm(user2)

print(unit_1)                              # [0.5 0.5 0.5 0.5]
print(unit_2)                              # [0.5 0.5 0.5 0.5] — identical after normalizing

print(np.linalg.norm(unit_1))             # 1.0 — magnitude always 1 after normalizing
print(np.linalg.norm(unit_2))             # 1.0

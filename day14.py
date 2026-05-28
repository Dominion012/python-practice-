import numpy as np
from scipy import stats

# =============================================
# Day 14 - Probability for Machine Learning
# =============================================


# --- Basic Probability ---

# probability = favourable outcomes / total outcomes
p_king     = 4 / 52          # probability of drawing a King
p_not_king = 1 - p_king      # probability of NOT drawing a King
print(p_king)
print(p_not_king)

# simulate 10000 card draws — should be close to 0.0769
cards = ["King"] * 4 + ["Other"] * 48
draws = np.random.choice(cards, size=10000)
p_king_sim = np.sum(draws == "King") / len(draws)
print(p_king_sim)


# --- Conditional Probability ---
# P(A | B) = P(A and B) / P(B)

# hospital dataset — P(diabetes | positive test)
p_diabetes     = 2000 / 10000
p_positive     = 3000 / 10000
p_both         = 1800 / 10000

p_diabetes_given_positive = p_both / p_positive
print(p_diabetes)                    # 0.20
print(p_positive)                    # 0.30
print(p_diabetes_given_positive)     # 0.60 — 60% chance of diabetes if test is positive


# --- Independent vs Dependent Events ---

# independent — rolling a 6 twice (one event doesn't affect the other)
p_six       = 1 / 6
p_two_sixes = p_six * p_six
print(p_two_sixes)   # 0.0278

# dependent — drawing 2 aces without replacement
p_ace_first  = 4 / 52
p_ace_second = 3 / 51    # one ace already removed
p_two_aces   = p_ace_first * p_ace_second
print(p_two_aces)    # 0.0045

# simulation — rolling two dice 10000 times
rolls = np.random.randint(1, 7, size=(10000, 2))
both_six = np.sum((rolls[:, 0] == 6) & (rolls[:, 1] == 6))
print(both_six / 10000)   # close to 0.0278


# --- Bayes Theorem ---
# P(A | B) = P(B | A) * P(A) / P(B)

# spam filter example
p_spam            = 0.30
p_free_given_spam = 0.60
p_free            = 0.20

p_spam_given_free = (p_free_given_spam * p_spam) / p_free
print(p_spam_given_free)   # 0.90 — 90% chance it's spam if it contains "free"

# factory defect example — base rate fallacy
p_defective         = 0.02
p_good              = 0.98
p_fail_given_defect = 0.95
p_fail_given_good   = 0.04

p_fail = p_fail_given_defect * p_defective + p_fail_given_good * p_good
p_defective_given_fail = (p_fail_given_defect * p_defective) / p_fail
print(p_fail)                    # total probability of failing test
print(p_defective_given_fail)    # ~0.32 — only 32% of failed phones are actually defective


# --- Probability Distributions ---

# uniform — every outcome equally likely (die rolls)
die_rolls = np.random.randint(1, 7, size=10000)
for i in range(1, 7):
    print(f"{i}: {np.sum(die_rolls == i) / 10000:.3f}")   # all close to 0.167

# binomial — probability of exactly k successes in n trials
# P(exactly 3 heads in 8 coin flips)
n, p = 8, 0.5
print(stats.binom.pmf(3, n, p))   # 0.219

# normal distribution
data = np.random.normal(loc=50, scale=10, size=1000)
print(np.mean(data))    # close to 50
print(np.std(data))     # close to 10


# --- Probability in ML Predictions ---

# sigmoid converts any number to a probability between 0 and 1
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# patient risk scores from a medical model
risk_scores  = np.array([-3, -1.5, 0, 0.5, 1.2, 2.5, 3.8])
probabilities = sigmoid(risk_scores)
print(probabilities)

# classify as high risk (1) or low risk (0) using threshold 0.5
high_risk = (probabilities >= 0.5).astype(int)
low_risk  = (probabilities < 0.5).astype(int)
print(high_risk)
print(f"High risk patients: {np.sum(high_risk)}")   # 5

# spam classification example
emails          = np.array([-1.2, 0.8, 2.1, -0.5, 1.9])
spam_probability = sigmoid(emails)
is_spam         = (spam_probability >= 0.5).astype(int)
print(spam_probability)
print(is_spam)   # 0 = not spam, 1 = spam

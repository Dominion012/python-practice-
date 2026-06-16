import numpy as np

def relu(z):
     return np.maximum(0, z)

   

def sigmoid(z):
     return 1 / (1 + np.exp(-z))



def forward_pass(x, W1, b1, W2, b2):
    z1 =  x @ W1 + b1      # hidden layer weighted sum:  x @ W1 + b1
    a1 = relu(z1)      # hidden layer activation:    relu(z1)
    z2 =  a1 @ W2 +b2        # output layer weighted sum:  a1 @ W2 + b2
    output = sigmoid(z2)   # output activation:          sigmoid(z2)
    return output

def softmax(z):
     return (np.exp(z) / np.exp(z).sum())

# Worked example from before
x  = np.array([1, 2])
W1 = np.array([[0.5, -0.5],
               [0.3,  0.8]])
b1 = np.array([0.1, -0.2])
W2 = np.array([[0.4], [-0.6]])
b2 = np.array([0.05])

# print(forward_pass(x, W1, b1, W2, b2))  # expect ≈ 0.4975
z = np.array([2.0, 1.0, 0.1])

def bce (y_true, y_pred):
     return -(y_true * np.log(y_pred) + (1 - y_true)* np.log(1 - y_pred))

# print(bce(1, 0.4975))
# print(bce(0, 0.4975))

# ============================================================
# TOPIC 7: Full Neural Network From Scratch
# ============================================================

X_train = np.array([[0.1, 0.2],
                    [0.9, 0.8],
                    [0.2, 0.1],
                    [0.8, 0.9]])
y_train = np.array([[0], [1], [0], [1]])

np.random.seed(42)
W1 = np.random.randn(2, 3) * 0.1
b1 = np.zeros((1, 3))
W2 = np.random.randn(3, 1) * 0.1
b2 = np.zeros((1, 1))

lr = 0.1
epochs = 1000

for epoch in range(epochs):
    # FORWARD PASS
    Z1 = X_train @ W1 + b1
    A1 = relu(Z1)
    Z2 = A1 @ W2 + b2
    output = sigmoid(Z2)

    loss = bce(y_train, output).mean()

    # BACKWARD PASS — fill in the 4 lines marked ???
    dZ2 = output - y_train                                    # error signal: output - y_train
    dW2 = A1.T @ dZ2                                    # gradient for W2: A1.T @ dZ2
    db2 = dZ2.sum(axis=0, keepdims=True)
    dA1 = dZ2 @ W2.T                            # pass error back through W2
    dZ1 = dA1 * (Z1 > 0)                                   # ReLU gate: dA1 * (Z1 > 0)
    dW1 = X_train.T @dZ1                                 # gradient for W1: X_train.T @ dZ1
    db1 = dZ1.sum(axis=0, keepdims=True)

    # UPDATE WEIGHTS
    W1 -= lr * dW1
    b1 -= lr * db1
    W2 -= lr * dW2
    b2 -= lr * db2

    if epoch % 200 == 0:
        print(f"Epoch {epoch:4d} | Loss: {loss:.4f}")

print("\nFinal predictions:")
for x, y in zip(X_train, y_train):
    pred = sigmoid(relu(x @ W1 + b1) @ W2 + b2)[0, 0]
    print(f"  Input {x} → predicted {pred:.3f}, true label {y[0]}")

# ============================================================
# TOPIC 8: Same Network in PyTorch
# ============================================================
import torch
import torch.nn as nn

X_t = torch.tensor([[0.1, 0.2], [0.9, 0.8], [0.2, 0.1], [0.8, 0.9]], dtype=torch.float32)
y_t = torch.tensor([[0], [1], [0], [1]], dtype=torch.float32)

model = nn.Sequential(
    nn.Linear(2, 3),   # W1, b1 (same as your NumPy W1)
    nn.ReLU(),
    nn.Linear(3, 1),   # W2, b2
    nn.Sigmoid()
)

criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

for epoch in range(1000):
    output = model(X_t)
    loss = criterion(output, y_t)

    optimizer.zero_grad()   # clear old gradients (PyTorch accumulates by default)
    loss.backward()         # backprop — replaces all 7 of your dZ2/dW2/dZ1/dW1 lines
    optimizer.step()        # weight update — replaces your 4 W -= lr*dW lines

    if epoch % 200 == 0:
        print(f"Epoch {epoch:4d} | Loss: {loss.item():.4f}")

print("\nFinal predictions (PyTorch):")
with torch.no_grad():
    preds = model(X_t)
    for x, pred, label in zip(X_t, preds, y_t):
        print(f"  Input {x.tolist()} → predicted {pred.item():.3f}, true label {int(label.item())}")

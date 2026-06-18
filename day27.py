# Day 27 — Deep Learning with PyTorch
# Week 6 Day 2

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

# ============================================================
# TOPIC 2: Tensors
# ============================================================
# PyTorch's version of NumPy arrays. Key extras: GPU support
# and gradient tracking (requires_grad) — how autograd works.

a = torch.tensor([1.0, 2.0, 3.0])   # from list
b = torch.zeros(3, 4)               # 3x4 of zeros
c = torch.randn(2, 3)               # random normal, shape (2,3)

# @ operator works on tensors exactly like NumPy
x = torch.randn(3, 2)
w = torch.randn(2, 4)
result = x @ w                       # (3,2) @ (2,4) → (3,4)
print(f"Tensor matmul shape: {result.shape} | dtype: {result.dtype}")

# NumPy ↔ Tensor (shares memory — changes to one affect the other)
arr  = np.array([1.0, 2.0])
t    = torch.from_numpy(arr)
back = t.numpy()

# ============================================================
# TOPIC 3: nn.Module — Custom Network Architecture
# ============================================================
# Base class for all PyTorch models.
# __init__  → define layers
# forward() → define how data flows through them

class TitanicNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1     = nn.Linear(6, 16)   # 6 input features → 16 hidden neurons
        self.fc2     = nn.Linear(16, 8)   # 16 → 8
        self.fc3     = nn.Linear(8, 1)    # 8 → 1 (binary output)
        self.relu    = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

model = TitanicNet()
total_params = sum(p.numel() for p in model.parameters())
print(f"\nModel architecture:\n{model}")
print(f"Total parameters: {total_params}")   # 112 + 136 + 9 = 257

# ============================================================
# TOPIC 4: Dataset & DataLoader
# ============================================================
# Dataset  → makes data indexable (dataset[i] returns one sample)
# DataLoader → batches + shuffles + can load in parallel
# shuffle=True prevents model learning row order instead of patterns
# 100 samples ÷ batch_size=32 → 3 full batches + 1 partial = 4 batches

class TitanicDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ============================================================
# TOPIC 5: Training Loop — The Standard 5-Step Pattern
# ============================================================
# Every PyTorch training iteration follows this exact order:
#
#   optimizer.zero_grad()     1. clear old gradients (PyTorch accumulates by default)
#   output = model(X_batch)   2. forward pass
#   loss = criterion(output)  3. compute loss
#   loss.backward()           4. backprop — computes all gradients automatically
#   optimizer.step()          5. update weights using those gradients

# ============================================================
# TOPIC 6: Optimizers — SGD vs Adam
# ============================================================
# SGD:  W = W - lr × gradient  (same lr for every weight, every step)
# Adam: adapts lr per weight based on gradient history → faster convergence
#
# Use Adam lr=0.001 as the default for most problems (tabular, NLP, vision).
# Lower lr than SGD because Adam internally amplifies gradients — 0.01 would overshoot.
# Switch to SGD only when following a paper that specifically used it.

# ============================================================
# TOPIC 7: Saving & Loading Models
# ============================================================
# Save state_dict (weights dict), NOT the whole model object.
# Whole-model saves break across Python versions and file paths.
#
#   torch.save(model.state_dict(), 'model.pth')
#
#   model = TitanicNet()
#   model.load_state_dict(torch.load('model.pth', weights_only=True))
#   model.eval()    ← always call before inference (disables dropout/batchnorm randomness)

# ============================================================
# TOPIC 8: Titanic Classifier — Full Pipeline
# ============================================================

# 1. Load & Preprocess
df = pd.read_csv('tested.csv')
df['Sex'] = (df['Sex'] == 'male').astype(int)        # male=1, female=0
df['Age']  = df['Age'].fillna(df['Age'].median())
df['Fare'] = df['Fare'].fillna(df['Fare'].median())

features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
X = df[features].values
y = df['Survived'].values.reshape(-1, 1)

X = StandardScaler().fit_transform(X)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. DataLoaders
train_loader = DataLoader(TitanicDataset(X_train, y_train), batch_size=32, shuffle=True)
val_loader   = DataLoader(TitanicDataset(X_val,   y_val),   batch_size=32)

# 3. Model, Loss, Optimizer
model     = TitanicNet()
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 4. Training Loop (5 steps per batch)
for epoch in range(100):
    model.train()
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        output = model(X_batch)
        loss   = criterion(output, y_batch)
        loss.backward()
        optimizer.step()

    if epoch % 20 == 0:
        model.eval()
        with torch.no_grad():
            val_preds = model(torch.tensor(X_val, dtype=torch.float32))
            val_loss  = criterion(val_preds, torch.tensor(y_val, dtype=torch.float32))
            accuracy  = ((val_preds > 0.5) == torch.tensor(y_val, dtype=torch.bool)).float().mean()
        print(f"Epoch {epoch:3d} | Val Loss: {val_loss:.4f} | Val Accuracy: {accuracy:.3f}")

# 5. Save
torch.save(model.state_dict(), 'titanic_model.pth')
print("\nModel saved → titanic_model.pth")

# NOTE: 100% val accuracy after epoch 20 = overfitting on a tiny val set (84 samples).
# Strong features (Sex, Pclass) + no regularization = model memorizes the split.
# Fix: add nn.Dropout(0.3) between layers and stop training when val loss plateaus.

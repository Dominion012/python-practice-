# Day 28 — Convolutional Neural Networks (CNNs)
# Week 6 Day 3

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# ============================================================
# TOPIC 2: Convolutions — Kernels and Filters
# ============================================================
# A kernel slides across the image; at each position it does an elementwise
# multiply with the patch underneath, then sums into one output pixel.
# Same w·x+b operation as a neuron — just reused (slid) across every
# position instead of getting brand-new weights each time.

image_patch = np.array([[0, 2, 1],
                         [1, 1, 0],
                         [0, 2, 1]])

kernel = np.array([[1, 0, -1],
                    [1, 0, -1],
                    [1, 0, -1]])

output_pixel = (image_patch * kernel).sum()
print(f"Convolution output: {output_pixel}")   # -1

# ============================================================
# TOPIC 3: Pooling — Shrinking Feature Maps
# ============================================================
# Max pooling slides a window and keeps only the largest value in each —
# shrinks the data and adds tolerance to small pixel shifts. Zero learnable
# parameters, same as ReLU — a fixed operation, not something trained.

feature_map = torch.tensor([[1., 3., 2., 4.],
                             [5., 2., 1., 0.],
                             [0., 1., 6., 2.],
                             [3., 0., 2., 1.]])

pool = nn.MaxPool2d(kernel_size=2, stride=2)
pooled = pool(feature_map.unsqueeze(0).unsqueeze(0))   # add batch + channel dims
print(f"Pooled output:\n{pooled}")

# ============================================================
# TOPIC 4: Building a CNN with nn.Module
# ============================================================
# Conv2d learns out_channels independent filters at once — each produces
# its own feature map. Stack Conv2d -> ReLU -> MaxPool, then flatten into
# regular Linear layers (same kind built in Day 26/27).

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        self.pool  = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu  = nn.ReLU()
        self.fc1   = nn.Linear(16 * 7 * 7, 32)   # after 2 pools: 28->14->7
        self.fc2   = nn.Linear(32, 10)           # 10 digit classes (0-9)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))   # (batch, 8, 14, 14)
        x = self.pool(self.relu(self.conv2(x)))   # (batch, 16, 7, 7)
        x = x.view(x.size(0), -1)                 # flatten -> (batch, 16*7*7)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)                            # raw logits — softmax applied by CrossEntropyLoss
        return x

model = SimpleCNN()
print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters())}")   # 26,698

dummy_batch = torch.randn(4, 1, 28, 28)             # batch of 4 fake 28x28 images
print(f"Output shape: {model(dummy_batch).shape}")  # (4, 10) — one score per class, per image

# ============================================================
# TOPIC 5: Feature Maps — What Each Layer "Sees"
# ============================================================
# out_channels=8 means 8 independent learned filters, each producing a
# different feature map. Confirmed below: each channel's stats differ —
# proof they're each detecting something different, not duplicates.

dummy_image = torch.randn(1, 1, 28, 28)
feature_maps = model.relu(model.conv1(dummy_image))   # shape (1, 8, 28, 28)

for ch in range(8):
    channel_map = feature_maps[0, ch]
    print(f"Channel {ch}: mean={channel_map.mean():.3f}, max={channel_map.max():.3f}")

# ============================================================
# TOPIC 6: Training the CNN — Same 5-Step Loop as Day 27
# ============================================================
# CrossEntropyLoss (multi-class) instead of BCELoss (binary): expects raw
# logits + integer class labels, applies softmax internally.

model     = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

dummy_X = torch.randn(8, 1, 28, 28)
dummy_y = torch.randint(0, 10, (8,))         # integer labels, NOT one-hot

print()
for epoch in range(5):
    optimizer.zero_grad()              # 1. clear old gradients
    output = model(dummy_X)            # 2. forward pass
    loss   = criterion(output, dummy_y)  # 3. compute loss
    loss.backward()                    # 4. backprop
    optimizer.step()                   # 5. update weights
    print(f"Epoch {epoch}: loss={loss.item():.4f}")

# ============================================================
# TOPIC 7: Data Augmentation & Transfer Learning
# ============================================================
# Augmentation: randomly transform images each epoch so the network can't
# just memorize pixels. NOTE: never flip digits horizontally — a flipped
# "6" isn't a valid digit. Augmentation must match what's realistic for the data.
#
# Transfer learning: freeze a network pretrained on 1.4M images (ResNet),
# swap in a new final layer for your task — reuses millions of already-
# learned general-purpose filters, you only train the new layer.

train_transform = transforms.Compose([
    transforms.RandomRotation(10),    # rotate up to +/-10 degrees — safe for digits
    transforms.ToTensor(),
])

resnet = models.resnet18(weights='IMAGENET1K_V1')
for param in resnet.parameters():
    param.requires_grad = False                        # freeze everything
resnet.fc = nn.Linear(resnet.fc.in_features, 10)        # only this layer trains

trainable = sum(p.numel() for p in resnet.parameters() if p.requires_grad)
total     = sum(p.numel() for p in resnet.parameters())
print(f"\nResNet — Trainable: {trainable:,} / Total: {total:,} ({trainable/total:.4%})")

# ============================================================
# TOPIC 8: MNIST CNN Classifier — Full Capstone
# ============================================================

# 1. Load real MNIST data (downloads ~25MB on first run)
mnist_transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root='./mnist_data', train=True,  download=True, transform=mnist_transform)
val_dataset   = datasets.MNIST(root='./mnist_data', train=False, download=True, transform=mnist_transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=64)

# 2. Model, Loss, Optimizer
model     = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 3. Training Loop
print()
for epoch in range(3):
    model.train()
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            preds = model(X_batch).argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)
    print(f"Epoch {epoch} | Val Accuracy: {correct/total:.4f}")

# 4. Save
torch.save(model.state_dict(), 'mnist_cnn.pth')
print("\nModel saved -> mnist_cnn.pth")

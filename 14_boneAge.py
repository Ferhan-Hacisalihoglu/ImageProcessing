# import libraries
import os
import cv2
import math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
from torch import nn, optim
import torch.nn.functional as F

# hyper parameters

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 48
IMAGE_SIZE = 256
EPOCHS = 5
LEARNING_RATE = 1e-4

# faster convolutions for fixed input size
torch.backends.cudnn.benchmark = True

# load data and preprocess
df = pd.read_csv("data/boneage/boneage-training-dataset.csv")
df["path"] = df["id"].apply(lambda x: os.path.join("data/boneage/boneage-training-dataset", f"{x}.png"))
df = df[df["path"].apply(os.path.exists)].reset_index(drop=True)
df["boneage"] = df["boneage"] / 240

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# read and resize every image once, then cache as uint8 array
def load_all(paths):
    arr = np.zeros((len(paths), IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    for i, p in enumerate(paths):
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        arr[i] = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    return arr

train_cache = f"data/boneage/cache_train_{IMAGE_SIZE}.npy"
test_cache = f"data/boneage/cache_test_{IMAGE_SIZE}.npy"

if os.path.exists(train_cache) and os.path.exists(test_cache):
    train_x = np.load(train_cache)
    test_x = np.load(test_cache)
else:
    train_x = load_all(train_df["path"].values)
    test_x = load_all(test_df["path"].values)
    np.save(train_cache, train_x)
    np.save(test_cache, test_x)

# keep whole dataset on the device as uint8, (N, 1, H, W)
train_x = torch.from_numpy(train_x).unsqueeze(1).to(DEVICE)
test_x = torch.from_numpy(test_x).unsqueeze(1).to(DEVICE)
train_y = torch.tensor(train_df["boneage"].values.astype("float32"), device=DEVICE)
test_y = torch.tensor(test_df["boneage"].values.astype("float32"), device=DEVICE)

# random affine on a whole batch at once (degrees=10, translate=0.1, scale=0.9-1.1)
def random_affine(x):
    B = x.size(0)
    angle = (torch.rand(B, device=x.device) * 2 - 1) * 10 * math.pi / 180
    scale = 0.9 + 0.2 * torch.rand(B, device=x.device)
    tx = (torch.rand(B, device=x.device) * 2 - 1) * 0.2
    ty = (torch.rand(B, device=x.device) * 2 - 1) * 0.2

    cos = torch.cos(angle) / scale
    sin = torch.sin(angle) / scale
    theta = torch.stack([
        torch.stack([cos, -sin, tx], dim=1),
        torch.stack([sin, cos, ty], dim=1),
    ], dim=1)  # (B, 2, 3)

    grid = F.affine_grid(theta, x.size(), align_corners=False)
    return F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)

# scale to [0, 1], augment (train only), normalize
def get_batch(x, y, idx, train):
    imgs = x[idx].float() / 255.0
    if train:
        imgs = random_affine(imgs)
    imgs = (imgs - 0.5) / 0.5
    imgs = imgs.contiguous(memory_format=torch.channels_last)
    return imgs, y[idx]

# define cnn model

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        
        # conv1
        self.conv1_1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        self.bn1_1 = nn.BatchNorm2d(32)
        self.relu1_1 = nn.ReLU()

        self.conv1_2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        self.bn1_2 = nn.BatchNorm2d(32)
        self.relu1_2 = nn.ReLU()
        
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        # conv2    
        self.conv2_1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.bn2_1 = nn.BatchNorm2d(64)
        self.relu2_1 = nn.ReLU()

        self.conv2_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.bn2_2 = nn.BatchNorm2d(64)
        self.relu2_2 = nn.ReLU()

        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        # fc
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=64 * 4 * 4, out_features=64)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=64, out_features=1)

    def forward(self, x):
        x = self.conv1_1(x)
        x = self.bn1_1(x)
        x = self.relu1_1(x)

        x = self.conv1_2(x)
        x = self.bn1_2(x)
        x = self.relu1_2(x)

        x = self.pool1(x)

        x = self.conv2_1(x)
        x = self.bn2_1(x)
        x = self.relu2_1(x)

        x = self.conv2_2(x)
        x = self.bn2_2(x)
        x = self.relu2_2(x)

        x = self.pool2(x)

        x = self.adaptive_pool(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)

        return x

# training loop
model = CNNModel().to(DEVICE).to(memory_format=torch.channels_last)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# mixed precision (only active on cuda)
use_amp = DEVICE.type == "cuda"
scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

n_train = train_x.size(0)

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    perm = torch.randperm(n_train, device=DEVICE)
    for i in range(0, n_train, BATCH_SIZE):
        idx = perm[i:i + BATCH_SIZE]
        imgs, labels = get_batch(train_x, train_y, idx, train=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=use_amp):
            outputs = model(imgs).squeeze(1)   # (B, 1) -> (B)
            loss = criterion(outputs.float(), labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        train_loss += loss.item() * idx.size(0)

    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {train_loss/n_train:.5f}")

# evaluate model
model.eval()
total_mse = 0.0
total_mae_months = 0.0
all_preds = []
all_labels = []

n_test = test_x.size(0)

with torch.no_grad():
    for i in range(0, n_test, BATCH_SIZE):
        idx = torch.arange(i, min(i + BATCH_SIZE, n_test), device=DEVICE)
        imgs, labels = get_batch(test_x, test_y, idx, train=False)

        with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=use_amp):
            outputs = model(imgs).squeeze(1)
        outputs = outputs.float()

        total_mse += criterion(outputs, labels).item() * idx.size(0)
        total_mae_months += (outputs - labels).abs().sum().item() * 240

        all_preds.append(outputs.cpu())
        all_labels.append(labels.cpu())

test_mse = total_mse / n_test
test_mae_months = total_mae_months / n_test

print(f"Test MSE (normalize): {test_mse:.5f}")
print(f"Test MAE: {test_mae_months:.2f} months ({test_mae_months/12:.2f} years)")

# est MSE (normalize): 0.02571
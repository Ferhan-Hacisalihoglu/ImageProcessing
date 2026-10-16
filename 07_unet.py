# import libraries

import os
from pathlib import Path
import copy
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch.nn.functional as F
import torchvision.transforms.functional as TF

# hyperparameters

CLASS_COLORS = np.array([
    [0x3C, 0x10, 0x98],  # building
    [0x84, 0x29, 0xF6],  # land
    [0x6E, 0xC1, 0xE4],  # road
    [0xFE, 0xDD, 0x3A],  # vegetation
    [0xE2, 0xA9, 0x29],  # water
    [0x9B, 0x9B, 0x9B],  # unlabeled
])
NUM_CLASSES = len(CLASS_COLORS)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = (512, 512)
BATCH_SIZE = 4
LEARNING_RATE = 0.001
EPOCHS = 50


def find_dataset_root():
    base = Path(__file__).resolve().parent
    candidates = [
        base / "data" / "aerial_dataset",
        base / "aerial_dataset",
        Path.cwd() / "aerial_dataset",
        Path.cwd() / "data" / "aerial_dataset",
    ]
    for c in candidates:
        if c.is_dir():
            return c
    raise FileNotFoundError("Not found aerial_dataset . Try these paths:\n" +
                            "\n".join(str(c) for c in candidates))


DATASET_ROOT = find_dataset_root()

# load dataset and preprocess images 

train_image_transforms = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),          
])

test_image_transforms = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),          
])

train_mask_transforms = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),          
])

test_mask_transforms = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),          
])

def rgb_to_class(arr):
    out = np.full(arr.shape[:2], 5, dtype=np.int64)
    for i, c in enumerate(CLASS_COLORS):
        out[(arr == c).all(axis=-1)] = i
    return out

class AerialDataset(Dataset):
    def __init__(self, samples, image_transform=None, mask_transform=None, train=False):
        self.samples = samples
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.train = train

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        img_p, mask_p = self.samples[i]
        image = Image.open(img_p).convert("RGB").resize(IMAGE_SIZE[::-1], Image.BILINEAR)
        mask = Image.open(mask_p).convert("RGB").resize(IMAGE_SIZE[::-1], Image.NEAREST)

        if self.train:
            if random.random() < 0.5:
                image, mask = TF.vflip(image), TF.vflip(mask)
            k = random.randint(0, 3)
            image, mask = TF.rotate(image, 90 * k), TF.rotate(mask, 90 * k)

        seed = random.randint(0, 2**31 - 1)
        torch.manual_seed(seed)
        image = self.image_transform(image)
        torch.manual_seed(seed)
        mask = self.mask_transform(mask)

        mask = (mask * 255).round().byte().permute(1, 2, 0).numpy()
        mask = torch.from_numpy(rgb_to_class(mask))   # (H, W) long
        return image, mask


def collect_samples(dataset_root):
    samples = []
    for tile in sorted(p for p in dataset_root.iterdir() if p.is_dir()):
        img_dir, mask_dir = tile / "images", tile / "masks"
        if not img_dir.is_dir():
            continue
        for image_path in sorted(img_dir.iterdir()):
            if image_path.suffix.lower() != ".jpg":
                continue
            mask_path = mask_dir / f"{image_path.stem}.png"
            if not mask_path.is_file():
                continue
            samples.append((image_path, mask_path))

    if not samples:
        raise FileNotFoundError(f"No aerial image/mask pairs found in {dataset_root}")
    return samples


samples = collect_samples(DATASET_ROOT)

train_samples, val_samples = train_test_split(samples, test_size=0.125, random_state=42)

train_dataset = AerialDataset(train_samples, train_image_transforms, train_mask_transforms, train=True)
val_dataset = AerialDataset(val_samples, test_image_transforms, test_mask_transforms, train=False)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# define U-Net model architecture 

class QuadConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNetModel(nn.Module):
    def __init__(self, in_channels=3, out_channels=NUM_CLASSES):
        super().__init__()

        # encoder
        self.enc1 = QuadConv(in_channels, 16)
        self.enc2 = QuadConv(16, 32)
        self.enc3 = QuadConv(32, 64)
        self.enc4 = QuadConv(64, 128)
        self.enc5 = QuadConv(128, 256)
        self.pool = nn.MaxPool2d(2)

        # bottleneck
        self.bottleneck = QuadConv(256, 512)

        # decoder
        self.up6 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec6 = QuadConv(256 + 256, 256)

        self.up7 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec7 = QuadConv(128 + 128, 128)

        self.up8 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec8 = QuadConv(64 + 64, 64)

        self.up9 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec9 = QuadConv(32 + 32, 32)

        self.up10 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec10 = QuadConv(16 + 16, 16)

        self.out_conv = nn.Conv2d(16, out_channels, kernel_size=1)

    def forward(self, x):
        c1 = self.enc1(x)               # 512x512
        c2 = self.enc2(self.pool(c1))   # 256x256
        c3 = self.enc3(self.pool(c2))   # 128x128
        c4 = self.enc4(self.pool(c3))   # 64x64
        c5 = self.enc5(self.pool(c4))   # 32x32
        c6 = self.bottleneck(self.pool(c5))  # 16x16

        c7 = self.dec6(torch.cat([self.up6(c6), c5], dim=1))    # 16 -> 32
        c8 = self.dec7(torch.cat([self.up7(c7), c4], dim=1))    # 32 -> 64
        c9 = self.dec8(torch.cat([self.up8(c8), c3], dim=1))    # 64 -> 128
        c10 = self.dec9(torch.cat([self.up9(c9), c2], dim=1))   # 128 -> 256
        c11 = self.dec10(torch.cat([self.up10(c10), c1], dim=1))  # 256 -> 512

        return self.out_conv(c11)

# define callback functions

class EarlyStopping: 
    
    def __init__(self,patience):

        self.patience = patience
        self.best_loss = np.inf
        self.best_model = None
        self.counter = 0

    def __call__(self, loss, model):
        
        if loss < self.best_loss: 
            self.best_loss = loss
            self.best_model = copy.deepcopy(model.state_dict())
            self.counter = 0 
        else: 
            self.counter += 1

        if self.counter >= self.patience:
            return True 
        
        return False

# train the model 

ce_loss = nn.CrossEntropyLoss()


def dice_loss(logits, target, eps=1e-6):
    probs = logits.softmax(1)
    onehot = F.one_hot(target, NUM_CLASSES).permute(0, 3, 1, 2).float()
    inter = (probs * onehot).sum((0, 2, 3))
    denom = probs.sum((0, 2, 3)) + onehot.sum((0, 2, 3))
    return 1 - ((2 * inter + eps) / (denom + eps)).mean()


def criterion(logits, target):
    return ce_loss(logits, target) + dice_loss(logits, target)


model = UNetModel().to(DEVICE)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=5)
early_stopping = EarlyStopping(patience=10)  


def run_epoch(loader, train):
    model.train(train)
    total, n = 0.0, 0
    inter = torch.zeros(NUM_CLASSES)
    union = torch.zeros(NUM_CLASSES)
    with torch.set_grad_enabled(train):
        for images, masks in loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            if train:
                optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            if train:
                loss.backward()
                optimizer.step()
            total += loss.item() * images.size(0)
            n += images.size(0)

            pred = outputs.argmax(1)
            for c in range(NUM_CLASSES):
                inter[c] += ((pred == c) & (masks == c)).sum().item()
                union[c] += ((pred == c) | (masks == c)).sum().item()

    valid = union > 0
    miou = (inter[valid] / union[valid]).mean().item()
    return total / n, miou


for epoch in range(EPOCHS):
    train_loss, train_miou = run_epoch(train_loader, train=True)
    val_loss, val_miou = run_epoch(val_loader, train=False)

    print(f"Epoch [{epoch + 1}/{EPOCHS}] loss: {train_loss:.5f} - mIoU: {train_miou:.4f} - val_loss: {val_loss:.5f} - val_mIoU: {val_miou:.4f}")

    scheduler.step(val_loss)

    if early_stopping(val_loss, model):
        print(f"Early stopping at epoch {epoch + 1}")
        break

os.makedirs("models", exist_ok=True)
torch.save(early_stopping.best_model, 'models/05_unet.pth')

# visualize predictions

model.load_state_dict(torch.load('models/05_unet.pth'))


def colorize(class_map):
    return CLASS_COLORS[class_map.numpy()].astype("uint8")


def show_prediction():
    model.eval()
    indices = random.sample(range(len(val_dataset)), 9)

    fig, axs = plt.subplots(3, 3, figsize=(14, 14))
    subfigs = fig.subfigures(3, 3) if False else None

    plt.close(fig)
    fig = plt.figure(figsize=(18, 12), constrained_layout=True)
    subfigs = fig.subfigures(3, 3)

    for subfig, idx in zip(subfigs.flat, indices):
        image, mask_true = val_dataset[idx]
        with torch.no_grad():
            pred = model(image.unsqueeze(0).to(DEVICE)).argmax(1).cpu()[0]

        subfig.suptitle(f"Örnek #{idx}", fontsize=11)
        axs = subfig.subplots(1, 3)

        axs[0].imshow(image.permute(1, 2, 0).numpy())
        axs[0].set_title("Input")

        axs[1].imshow(colorize(mask_true))
        axs[1].set_title("Truth")

        axs[2].imshow(colorize(pred))
        axs[2].set_title("Prediction")

        for ax in axs:
            ax.axis("off")

    plt.show()

show_prediction()

# val_mIoU: 0.5183
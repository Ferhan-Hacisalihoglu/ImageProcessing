# import libraries
import torch
from torch import optim, nn

import torchvision
from torchvision import transforms, datasets

from matplotlib import pyplot as plt
import numpy as np

# hyperparameters

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
NOISE_DIM = 100
IMAGE_SIZE = ( 28,28,1)
LEARNING_RATE = 2e-4
EPOCHS = 20

# load dataset and preprocess data

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5))
])

train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=train_transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# define the gan models

class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc = nn.Linear(NOISE_DIM, 7*7*256, bias=False)   
        self.bn1 = nn.BatchNorm1d(7*7*256)                    
        self.lRelu1 = nn.LeakyReLU()

        self.conv1 = nn.ConvTranspose2d(256, 128, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.lRelu2 = nn.LeakyReLU()

        self.conv2 = nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.lRelu3 = nn.LeakyReLU()

        self.conv3 = nn.ConvTranspose2d(64, 1, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False)
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        x = self.fc(x)
        x = self.bn1(x)
        x = self.lRelu1(x)

        x = x.view(-1, 256, 7, 7) 

        x = self.conv1(x)
        x = self.bn2(x)
        x = self.lRelu2(x)

        x = self.conv2(x)
        x = self.bn3(x)
        x = self.lRelu3(x)

        x = self.conv3(x)
        x = self.tanh(x)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=2, padding=2)
        self.act1 = nn.LeakyReLU(0.3)
        self.drop1 = nn.Dropout(0.3)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)
        self.act2 = nn.LeakyReLU(0.3)
        self.drop2 = nn.Dropout(0.3)

        self.flatten = nn.Flatten()              
        self.fc = nn.Linear(128*7*7, 1)          

    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        x = self.drop1(x)

        x = self.conv2(x)
        x = self.act2(x)
        x = self.drop2(x)

        x = self.flatten(x)
        x = self.fc(x)
        return x
        
# train the gan models
generator = Generator().to(DEVICE)
discriminator = Discriminator().to(DEVICE)

criterion = nn.BCEWithLogitsLoss()
optimizer_G = optim.Adam(generator.parameters(), lr=LEARNING_RATE,betas=(0.6, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE,betas=(0.6, 0.999))

for epoch in range(EPOCHS):
    d_loss_total = 0.0
    g_loss_total = 0.0

    for real_imgs, _ in train_loader:
        real_imgs = real_imgs.to(DEVICE)
        batch_size = real_imgs.size(0)

        real_labels = torch.ones(batch_size, 1, device=DEVICE)
        fake_labels = torch.zeros(batch_size, 1, device=DEVICE)

        # discriminator train
        noise = torch.randn(batch_size, NOISE_DIM, device=DEVICE)
        fake_imgs = generator(noise)

        optimizer_D.zero_grad()
        d_loss_real = criterion(discriminator(real_imgs), real_labels)
        d_loss_fake = criterion(discriminator(fake_imgs.detach()), fake_labels)  
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        optimizer_D.step()

        # generator train
        optimizer_G.zero_grad()
        g_loss = criterion(discriminator(fake_imgs), real_labels)  
        g_loss.backward()
        optimizer_G.step()

        # calculate loss
        d_loss_total += d_loss.item()
        g_loss_total += g_loss.item()

    print(f"Epoch [{epoch+1}/{EPOCHS}] | D loss: {d_loss_total / len(train_loader):.5f} | G loss: {g_loss_total / len(train_loader):.5f}")

# evaluate the models

generator.eval()
with torch.no_grad():
    noise = torch.randn(20, NOISE_DIM, device=DEVICE)
    fake_imgs = generator(noise).cpu()

def denorm(x):
    return ((x + 1) / 2).clamp(0, 1)

fig, axes = plt.subplots(4, 5, figsize=(10, 8))

for i, ax in enumerate(axes.flat):
    ax.imshow(denorm(fake_imgs[i]).squeeze(), cmap="gray")
    ax.set_title(f"Generated #{i+1}", fontsize=12)
    ax.axis("off")

plt.tight_layout()
plt.show()
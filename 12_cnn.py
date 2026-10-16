# import libraries

import torch
from torch import nn,optim,utils
import torchvision
from torchvision import transforms,datasets

import matplotlib.pyplot as plt

# hyper parameters

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 256
EPOCHS = 10
LEARNING_RATE = 5e-4

# data load

train_transform = transforms.Compose([ 
    transforms.Resize((28, 28)),
    transforms.RandomAffine(degrees=10,translate=(0.1, 0.1),scale=(0.9, 1.1),),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: 1 - x),
    transforms.Normalize((0.5), (0.5))
])

test_transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: 1 - x),
    transforms.Normalize((0.5), (0.5))
])


train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=train_transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=test_transform)

train_loader = utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = utils.data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# visualize samples

"""
images, labels = next(iter(train_loader))

fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for i, ax in enumerate(axes.flat):
    ax.imshow(images[i].squeeze(), cmap="gray")
    ax.set_title(f"Label: {labels[i].item()}")
    ax.axis("off")

plt.tight_layout()
plt.show()
"""

# define model

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

        # fc
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=64 * 4 * 4, out_features=64)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=64, out_features=10)

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

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)

        return x

# train model

model = CNNModel().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

model.train()
for epoch in range(EPOCHS):
    running_loss = 0.0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(DEVICE), target.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)

    print(f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {avg_train_loss:.5f}")

torch.save(model, 'models/12_cnn.pth')

# evaluate model

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(DEVICE), target.to(DEVICE)

        outputs = model(data)

        preds = outputs.argmax(dim=1)
        correct += (preds == target).sum().item()
        total += target.size(0)

print(f"Test Accuracy: {100 * correct / total:.2f}%")

# accuracy : 99.40%
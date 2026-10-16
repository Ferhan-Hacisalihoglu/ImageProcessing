# import libraries
import torch
from torch import optim,nn
import torchvision
from torchvision import transforms
import torchvision.transforms.functional as TF

import matplotlib.pyplot as plt
import cv2
import numpy as np

# hiyperparameters

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 256
LEARNING_RATE = 0.001
EPOCHS = 10

# load MNIST dataset and preprocess

class CannyEdge:
    def __init__(self, low_threshold=50, high_threshold=150):
        super().__init__()

        self.low = low_threshold
        self.high = high_threshold

    def __call__(self, image):
        img = np.asarray(image, dtype=np.uint8)
        img_edges = cv2.Canny(img, self.low, self.high)
        return np.stack(img_edges, axis=-1)

transform = transforms.Compose([ 
    TF.equalize,
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    CannyEdge(low_threshold=50, high_threshold=150),
    transforms.ToTensor()
])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

train_x, train_y = next(iter(train_loader))
test_x, test_y = next(iter(test_loader))

# visualize the first 4 images and their corresponding edge maps

"""
fig, axes = plt.subplots(2, 2, figsize=(6, 6))
axes = axes.flat
for ax, (title,im) in zip(axes, zip(train_y[:4].numpy(), train_x[:4])):
    ax.imshow(im[0], cmap='gray')
    ax.set_title(f'Label: {title}')
    ax.axis('off')

plt.suptitle("MNIST Images Preprocessed States")
plt.tight_layout()
plt.show()
"""

# ann model creation

class MNISTModel(nn.Module):
    def __init__(self, input_size=28 * 28):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 128)   
        self.dropout = nn.Dropout(0.5)         
        self.fc2 = nn.Linear(128, 64)           
        self.fc3 = nn.Linear(64, 10)   
        self.relu = nn.ReLU()         

    def forward(self, x):
        x = x.view(x.size(0), -1)             
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        return self.fc3(x)                      

# ann model training

model = MNISTModel(input_size=train_x[0].numel()).to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss = nn.CrossEntropyLoss()

for epoch in range(EPOCHS):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss_value = loss(output, target)
        loss_value.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}/{EPOCHS}, Loss: {loss_value.item():.5f}')

# evaluate the model
model.eval()
correct_predictions = 0
total_samples = len(test_loader.dataset)

with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        pred = output.argmax(dim=1)
        correct_predictions += (pred == target).sum().item()

test_accuracy = 100.0 * correct_predictions / total_samples

print(f'Test Accuracy: {test_accuracy:.2f}')

# accuracy rate : % 93.78
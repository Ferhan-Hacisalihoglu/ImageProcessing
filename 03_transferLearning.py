# import libraries
import torch
from torch import optim,nn
import torchvision
from torchvision import transforms,datasets,models

import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# hiyperparameters
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
EPOCHS = 10
DATA_DIR = './data/chest_xray'
IMAGE_SIZE = (224, 224)

# load dataset and preprocess data

train_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(root=os.path.join(DATA_DIR, 'train'),transform=train_transform)
val_dataset = datasets.ImageFolder(root=os.path.join(DATA_DIR, 'val'),transform=test_transform)
test_dataset = datasets.ImageFolder(root=os.path.join(DATA_DIR, 'test'),transform=test_transform)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

"""
print(f"Train images: {len(train_dataset)}")
print(f"Validation images: {len(val_dataset)}")
print(f"Test images: {len(test_dataset)}")
print(f"Classes: {train_dataset.classes}")
"""

# visualize sample data

"""
images, labels = next(iter(train_loader))

fig, axes = plt.subplots(1, 4, figsize=(12, 4))
for image, label, axis in zip(images[:4], labels[:4], axes):
    axis.imshow(image.permute(1, 2, 0))
    axis.set_title(train_dataset.classes[label.item()])
    axis.axis("off")

fig.suptitle("Train Verisinden 4 Örnek")
plt.tight_layout()
plt.show()
"""

# load the model

base_model = models.densenet121(pretrained=True).to(DEVICE)

for parameter in base_model.parameters():
    parameter.requires_grad = False

for param in base_model.features.denseblock4.parameters():
    param.requires_grad = True

for param in base_model.features.norm5.parameters():
    param.requires_grad = True

base_model.classifier = nn.Linear(base_model.classifier.in_features,1).to(DEVICE)

# train the model

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW([
    {'params': [p for p in base_model.features.parameters() if p.requires_grad], 'lr': 1e-5},
    {'params': base_model.classifier.parameters(), 'lr': 1e-3}],)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=3)

for epoch in range(EPOCHS):
    base_model.train()
    train_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = base_model(images)
        loss = criterion(outputs.squeeze(1), labels.float())
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)

    base_model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = base_model(images)
            loss = criterion(outputs.squeeze(1), labels.float())
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)

    print(f"Epoch [{epoch + 1}/{EPOCHS}], Train Loss: {avg_train_loss:.5f}, Val Loss: {avg_val_loss:.5f}")

    scheduler.step(avg_val_loss)

# evaluate the model

base_model.eval()
accuracyRate = 0.0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = base_model(images)
        predicted = (torch.sigmoid(outputs.squeeze(1)) >= 0.5).long()
        accuracyRate += (predicted == labels).sum().item()

print(f"Test Accuracy: {accuracyRate / len(test_dataset) * 100:.2f}%")

# Test Accuracy: 87.82%
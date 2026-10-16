# import libraries
import torch
from torch import optim,nn
import torchvision
from torchvision import transforms,datasets

import matplotlib.pyplot as plt
import numpy as np
import copy

# hiyperparameters

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 48
IMG_SIZE = (128, 128)
LEARNING_RATE = 1e-3
EPOCHS = 25

# load dataset and preprocess data

train_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.Flowers102(root='./data', split='train', transform=train_transform, download=True)
val_dataset = datasets.Flowers102(root='./data', split='val', transform=train_transform, download=True)
full_train = torch.utils.data.ConcatDataset([train_dataset, val_dataset])
test_dataset = datasets.Flowers102( root='./data', split='test', transform=test_transform, download=True)

train_loader = torch.utils.data.DataLoader(dataset=full_train, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# visualize sample data

"""
fig = plt.figure(figsize=(10, 5))

for i , (image,label) in enumerate(train_loader):
    if i == 4:
        break
    ax = fig.add_subplot(1, 4, i+1)
    ax.imshow(image[0].permute(1, 2, 0))
    ax.set_title(f'Label: {label[0]}')
    ax.axis('off')

plt.tight_layout()
plt.show()
"""

# define cnn model architecture

class CNNModel(nn.Module):
    def __init__(self, num_classes=102):
        super(CNNModel, self).__init__()

        # conv1
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()

        # conv2
        self.conv2_1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2_1 = nn.BatchNorm2d(64)
        self.relu2_1 = nn.ReLU()

        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2_2 = nn.BatchNorm2d(64)
        self.relu2_2 = nn.ReLU()

        self.conv2_3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2_3 = nn.BatchNorm2d(64)
        self.relu2_3 = nn.ReLU()

        self.conv2_4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2_4 = nn.BatchNorm2d(64)
        self.relu2_4 = nn.ReLU()

        # pool1
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # conv3
        self.conv3_1 = nn.Conv2d(64, 96, kernel_size=3, stride=1, padding=1)
        self.bn3_1 = nn.BatchNorm2d(96)
        self.relu3_1 = nn.ReLU()

        self.conv3_2 = nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1)
        self.bn3_2 = nn.BatchNorm2d(96)
        self.relu3_2 = nn.ReLU()

        self.conv3_3 = nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1)
        self.bn3_3 = nn.BatchNorm2d(96)
        self.relu3_3 = nn.ReLU()

        self.conv3_4 = nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1)
        self.bn3_4 = nn.BatchNorm2d(96)
        self.relu3_4 = nn.ReLU()

        # pool2
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # conv4
        self.conv4_1 = nn.Conv2d(96, 128, kernel_size=3, stride=1, padding=1)
        self.bn4_1 = nn.BatchNorm2d(128)
        self.relu4_1 = nn.ReLU()

        self.conv4_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn4_2 = nn.BatchNorm2d(128)
        self.relu4_2 = nn.ReLU()

        self.conv4_3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn4_3 = nn.BatchNorm2d(128)
        self.relu4_3 = nn.ReLU()

        self.conv4_4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn4_4 = nn.BatchNorm2d(128)
        self.relu4_4 = nn.ReLU()

        # pool3
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # conv5
        self.conv5_1 = nn.Conv2d(128, 192, kernel_size=3, stride=1, padding=1)
        self.bn5_1 = nn.BatchNorm2d(192)
        self.relu5_1 = nn.ReLU()

        self.conv5_2 = nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1)
        self.bn5_2 = nn.BatchNorm2d(192)
        self.relu5_2 = nn.ReLU()

        self.conv5_3 = nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1)
        self.bn5_3 = nn.BatchNorm2d(192)
        self.relu5_3 = nn.ReLU()

        self.conv5_4 = nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1)
        self.bn5_4 = nn.BatchNorm2d(192)
        self.relu5_4 = nn.ReLU()

        # pool4
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # conv6
        self.conv6_1 = nn.Conv2d(192, 256, kernel_size=3, stride=1, padding=1)
        self.bn6_1 = nn.BatchNorm2d(256)
        self.relu6_1 = nn.ReLU()

        self.conv6_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn6_2 = nn.BatchNorm2d(256)
        self.relu6_2 = nn.ReLU()

        self.conv6_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn6_3 = nn.BatchNorm2d(256)
        self.relu6_3 = nn.ReLU()

        self.conv6_4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn6_4 = nn.BatchNorm2d(256)
        self.relu6_4 = nn.ReLU()

        # pool5
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        # foolly connected
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256*4*4, 256)
        self.dropout = nn.Dropout(0.25)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, inputs, debug=False):
        # conv 1
        conv1 = self.conv1(inputs)
        bn1 = self.bn1(conv1)
        relu1 = self.relu1(bn1)

        # conv 2
        conv2_1 = self.conv2_1(relu1)
        bn2_1 = self.bn2_1(conv2_1)
        relu2_1 = self.relu2_1(bn2_1)

        conv2_2 = self.conv2_2(relu2_1)
        bn2_2 = self.bn2_2(conv2_2)
        relu2_2 = self.relu2_2(bn2_2)

        conv2_3 = self.conv2_3(relu2_2)
        bn2_3 = self.bn2_3(conv2_3)
        relu2_3 = self.relu2_3(bn2_3)

        conv2_4 = self.conv2_4(relu2_3)
        bn2_4 = self.bn2_4(conv2_4)
        relu2_4 = self.relu2_4(bn2_4)

        # pool 1
        pool1 = self.pool1(relu2_4)

        # conv 3
        conv3_1 = self.conv3_1(pool1)
        bn3_1 = self.bn3_1(conv3_1)
        relu3_1 = self.relu3_1(bn3_1)

        conv3_2 = self.conv3_2(relu3_1)
        bn3_2 = self.bn3_2(conv3_2)
        relu3_2 = self.relu3_2(bn3_2)

        conv3_3 = self.conv3_3(relu3_2)
        bn3_3 = self.bn3_3(conv3_3)
        relu3_3 = self.relu3_3(bn3_3)

        conv3_4 = self.conv3_4(relu3_3)
        bn3_4 = self.bn3_4(conv3_4)
        relu3_4 = self.relu3_4(bn3_4)

        # pool 2
        pool2 = self.pool2(relu3_4)

        # conv 4
        conv4_1 = self.conv4_1(pool2)
        bn4_1 = self.bn4_1(conv4_1)
        relu4_1 = self.relu4_1(bn4_1)

        conv4_2 = self.conv4_2(relu4_1)
        bn4_2 = self.bn4_2(conv4_2)
        relu4_2 = self.relu4_2(bn4_2)

        conv4_3 = self.conv4_3(relu4_2)
        bn4_3 = self.bn4_3(conv4_3)
        relu4_3 = self.relu4_3(bn4_3)

        conv4_4 = self.conv4_4(relu4_3)
        bn4_4 = self.bn4_4(conv4_4)
        relu4_4 = self.relu4_4(bn4_4)

        # pool 3
        pool3 = self.pool3(relu4_4)

        # conv 5
        conv5_1 = self.conv5_1(pool3)
        bn5_1 = self.bn5_1(conv5_1)
        relu5_1 = self.relu5_1(bn5_1)

        conv5_2 = self.conv5_2(relu5_1)
        bn5_2 = self.bn5_2(conv5_2)
        relu5_2 = self.relu5_2(bn5_2)

        conv5_3 = self.conv5_3(relu5_2)
        bn5_3 = self.bn5_3(conv5_3)
        relu5_3 = self.relu5_3(bn5_3)

        conv5_4 = self.conv5_4(relu5_3)
        bn5_4 = self.bn5_4(conv5_4)
        relu5_4 = self.relu5_4(bn5_4)

        # pool 4
        pool4 = self.pool4(relu5_4)

        # conv 6
        conv6_1 = self.conv6_1(pool4)
        bn6_1 = self.bn6_1(conv6_1)
        relu6_1 = self.relu6_1(bn6_1)

        conv6_2 = self.conv6_2(relu6_1)
        bn6_2 = self.bn6_2(conv6_2)
        relu6_2 = self.relu6_2(bn6_2)

        conv6_3 = self.conv6_3(relu6_2)
        bn6_3 = self.bn6_3(conv6_3)
        relu6_3 = self.relu6_3(bn6_3)

        conv6_4 = self.conv6_4(relu6_3)
        bn6_4 = self.bn6_4(conv6_4)
        relu6_4 = self.relu6_4(bn6_4)

        # pool 5
        pool5 = self.pool5(relu6_4)

        # fc
        flatten = self.flatten(pool5)
        fc1 = self.dropout(self.fc1(flatten))
        fc2 = self.fc2(fc1)

        return(fc2)


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

model = CNNModel(num_classes=102).to(device)
early_stopping = EarlyStopping(patience=5)  
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=3)
batch_length = len(train_loader)

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    avg_train_loss = train_loss / batch_length

    if early_stopping(avg_train_loss, model):
        print(f"Early stopping at epoch {epoch + 1}")
        break

    print(f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {avg_train_loss:.5f}")

    scheduler.step(avg_train_loss)

torch.save(early_stopping.best_model, 'models/02_cnn.pth')

# evaluate the model on test data

model.load_state_dict(torch.load('models/02_cnn.pth'))
model.eval()
accuracyRate = 0.0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        accuracyRate += (predicted == labels).sum().item()

print(f"Test Accuracy: {accuracyRate / len(test_dataset) * 100:.2f}%")

# Test Accuracy: 40.84%
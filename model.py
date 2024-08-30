import torch
import torch.nn as nn
import torch.nn.init as init
import torchvision.models
from torch.nn import functional as F
import torch.optim as optim
import os
import numpy as np
import copy
import time
import torchvision.transforms as transforms

# Define a CNN model

class BrainTumorNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(16, 32, 5)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, 3)
        self.bn3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 128, 3)
        self.bn4 = nn.BatchNorm2d(128)

        # Calculate the size of the feature map after the conv and pool layers
        # Input size: 128x128
        # After conv1: (128 - 5 + 1) = 124 -> 124x124
        # After pool1: 124 / 2 = 62 -> 62x62
        # After conv2: (62 - 5 + 1) = 58 -> 58x58
        # After pool2: 58 / 2 = 29 -> 29x29
        # After conv3: (29 - 3 + 1) = 27 -> 27x27
        # After pool3: 27 / 2 = 13.5 -> 13x13 (rounding down)
        # After conv4: (13 - 3 + 1) = 11 -> 11x11
        # After pool4: 11 / 2 = 5.5 -> 5x5 (rounding down)

        self.fc1 = nn.Linear(128 * 5 * 5, 512)
        self.dropout1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(0.5)

        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))

        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        x = self.dropout1(x)

        x = F.relu(self.fc2(x))
        x = self.dropout2(x)

        x = F.relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))

        return x
    
    
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Device:", device)

model = BrainTumorNet()
model = model.to(device)
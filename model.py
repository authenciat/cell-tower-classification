import torch
import torch.nn as nn

class TowerClassifier(nn.Module):
    def __init__(self):
        super(TowerClassifier, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 4)  # 4 classes: guyed, lattice, monopole, water_tank
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)
        
        # Activation functions
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Convolutional layers with ReLU and pooling
        x = self.pool(self.relu(self.conv1(x)))  # 224x224 -> 112x112
        x = self.pool(self.relu(self.conv2(x)))  # 112x112 -> 56x56
        x = self.pool(self.relu(self.conv3(x)))  # 56x56 -> 28x28
        
        # Flatten
        x = x.view(-1, 128 * 28 * 28)
        
        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x 
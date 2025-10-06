import torch
import torch.nn as nn
import torch.nn.functional as F

class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1,6,(5,5))
        self.conv2 = nn.Conv2d(6,16,(5,5))
        self.pool = nn.MaxPool2d((5,5))
        self.fc1 = nn.Linear(256,128)
        self.fc2 = nn.Linear(128,64)
        self.fc3 = nn.Linear(64,10)

    def forward(self, x):
        x = F.relu(self.conv1(x))       # First convolution followed by
        x = torch.nn.MaxPool2d(x,(2,2))                # a relu activation and a max pooling#
        x = F.relu(self.conv2(x))
        x = torch.nn.MaxPool2d(x,(2,2))
        x = torch.flatten(x,1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
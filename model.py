import torch.nn as nn
import torch.nn.functional as F

class MnistCNN(nn.Module):
    def __init__(self, kernel_numbers):
        """
        kernel_numbers: list of 4 integers representing number of kernels in each conv layer
        """
        super(MnistCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, kernel_numbers[0], kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(kernel_numbers[0], kernel_numbers[1], kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(kernel_numbers[1], kernel_numbers[2], kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(kernel_numbers[2], kernel_numbers[3], kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(kernel_numbers[3] * 1 * 1, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = x.view(-1, self.conv4.out_channels * 1 * 1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x 
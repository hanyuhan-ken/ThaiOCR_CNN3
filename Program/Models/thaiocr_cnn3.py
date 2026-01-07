import torch
import torch.nn as nn
import torch.nn.functional as F

class ThaiOCR_CNN3(nn.Module):

    def __init__(self, num_classes):
        super(ThaiOCR_CNN3, self).__init__()

        # Block 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.01),

            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.01),

            nn.MaxPool2d(2, 2),        
            nn.Dropout2d(0.3)
        )

        # Block 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.01),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.01),

            nn.MaxPool2d(2, 2),        
            nn.Dropout2d(0.3)
        )

        # Block 3
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.01),

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.01),

            nn.MaxPool2d(2, 2),        
            nn.Dropout2d(0.3)
        )

        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.dropout_fc = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.conv1(x)   
        x = self.conv2(x)   
        x = self.conv3(x)   

        x = x.view(x.size(0), -1)

        x = F.leaky_relu(self.fc1(x), 0.01)
        x = self.dropout_fc(x)
        x = self.fc2(x)
        return x

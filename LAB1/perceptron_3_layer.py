import torch
import torch.nn as nn
import torch.nn.functional as F


class Perceptron_3_layer(nn.Module):
    def __init__(self, input_dim=28*28, hidden1=256, hidden2=128, num_classes=10):
        super(Perceptron_3_layer, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, num_classes)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = x.view(x.size(0), -1)

        # Hidden layer 1
        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        # Hidden layer 2
        x = F.relu(self.fc2(x))
        x = self.dropout(x)

        # Output layer
        out = self.fc3(x)
        return out

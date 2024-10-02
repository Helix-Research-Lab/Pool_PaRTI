import torch.nn as nn
import torch.nn.functional as F

from torch.optim import Adam

import numpy as np
import torch



class MultiLabelNN(nn.Module):
    def __init__(self, input_size, num_classes, dropout=0.2, slope_Leaky_ReLU=0.01):
        super(MultiLabelNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.slope_Leaky_ReLU = slope_Leaky_ReLU
        # Initialize the first layer
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_out', nonlinearity='leaky_relu')
        
        self.dropout1 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(512, 128)
        # Initialize the second layer
        nn.init.kaiming_normal_(self.fc2.weight, mode='fan_out', nonlinearity='leaky_relu')
        
        self.dropout2 = nn.Dropout(dropout)
        
        self.fc3 = nn.Linear(128, num_classes)
        # Initialize the third layer (considering sigmoid activation, this initialization is more arbitrary)
        nn.init.xavier_normal_(self.fc3.weight)
        
        # Example projection for a residual connection (not used directly here due to structural limitations)
        self.projection = nn.Linear(input_size, num_classes)
        nn.init.kaiming_normal_(self.projection.weight, mode='fan_out', nonlinearity='leaky_relu')

    def forward(self, x):
        # Apply projection at the beginning to match dimensionality for residual connection
        identity = self.projection(x)

        x = F.leaky_relu(self.fc1(x), negative_slope=self.slope_Leaky_ReLU)
        x = self.dropout1(x)
        x = F.leaky_relu(self.fc2(x), negative_slope=self.slope_Leaky_ReLU)
        x = self.dropout2(x)
        
        # No sigmoid here if using BCEWithLogitsLoss
        x = self.fc3(x)
        
        # Adding the processed input to the output
        x += identity
        
        return torch.sigmoid(x)





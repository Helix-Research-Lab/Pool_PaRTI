import torch
import torch.nn as nn
import torch.nn.functional as F

def assert_no_nan(tensor, message=""):
    assert not torch.isnan(tensor).any(), message

def log_stats(x, prefix=""):
    mean = x.mean().item()
    var = x.var().item()
    std = x.std().item()
    max_val = x.max(dim=-1).values
    min_val = x.min(dim=-1).values
    print(f"{prefix} Mean: {mean}, Variance: {var}, Std: {std}, max: {max_val}, min: {min_val}")

class InteractionPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.1, negative_slope=0.01):
        super(InteractionPredictor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.residual = nn.Linear(input_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.negative_slope = negative_slope
        
        # Initialize weights
        self._initialize_weights()

    def forward(self, x):

        x = x.float()  # If x is your input tensor
        assert_no_nan(x, "NaN detected in x")
        identity = self.residual(x).float()

        assert_no_nan(identity, "NaN detected in identity")
        
        out = F.leaky_relu(self.bn1(self.fc1(x)), negative_slope=self.negative_slope)
        assert_no_nan(out, "NaN detected in out after relu")
        
        out = self.dropout(out)
        assert_no_nan(out, "NaN detected in out after dropout")
        out = F.leaky_relu(self.fc2(out) + identity, negative_slope=self.negative_slope)  # Residual connection
        assert_no_nan(out, "NaN detected in out after residual connection")
        out = self.fc3(out)
        assert_no_nan(out, "NaN detected in out after fc3")
        
        return out.squeeze()  # Squeeze the output to remove any extra dimension

    def _initialize_weights(self):
        # Initialize weights using He initialization suitable for ReLU activation
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                m.weight = m.weight.float()
                m.bias = m.bias.float()
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                m.weight = m.weight.float()
                m.bias = m.bias.float()
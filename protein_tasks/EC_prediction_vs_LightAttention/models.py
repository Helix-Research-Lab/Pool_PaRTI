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


class ECPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=6, dropout_rate=0.1, negative_slope=0.01):
        super(ECPredictor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.negative_slope = negative_slope
        
        # Initialize weights
        self._initialize_weights()

    def forward(self, x):

        x = x.float()  # If x is your input tensor
        assert_no_nan(x, "NaN detected in x")
            
        out = F.leaky_relu(self.bn1(self.fc1(x)), negative_slope=self.negative_slope)
        assert_no_nan(out, "NaN detected in out after relu")
        
        out = self.dropout(out)
        assert_no_nan(out, "NaN detected in out after dropout")
        out = F.leaky_relu(self.fc2(out), negative_slope=self.negative_slope)  # Residual connection
        assert_no_nan(out, "NaN detected in out after residual connection")
        out = self.dropout(out)
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

class ECPredictorLA(nn.Module):
    def __init__(self, input_dim=32, hidden_dim=32, output_dim=6, dropout_rate=0.1, negative_slope=0.01):
        super(ECPredictorLA, self).__init__()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.negative_slope = negative_slope
        
        # Initialize weights
        self._initialize_weights()

    def forward(self, x):

        x = x.float()  # If x is your input tensor
       
        assert_no_nan(x, "NaN detected in identity")
        
        out = F.leaky_relu(self.bn2(self.fc2(x)), negative_slope=self.negative_slope)
        assert_no_nan(out, "NaN detected in out after relu")
        out = self.dropout(out)
        out = self.fc3(out)
        assert_no_nan(out, "NaN detected in out after fc3")
        
        #return out.squeeze()  # Squeeze the output to remove any extra dimension

        return out

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



class LightAttention(nn.Module):
    def __init__(self, embeddings_dim=1280, dropout=0.25, kernel_size=9, conv_dropout: float = 0.25):
        super(LightAttention, self).__init__()

        """
        self.feature_convolution = nn.Conv1d(embeddings_dim, embeddings_dim, kernel_size, stride=1,
                                             padding=kernel_size // 2)
        self.attention_convolution = nn.Conv1d(embeddings_dim, embeddings_dim, kernel_size, stride=1,
                                               padding=kernel_size // 2)
                                               """

        self.feature_convolution = nn.Conv1d(embeddings_dim, embeddings_dim, kernel_size, stride=1, padding="same")
        self.attention_convolution = nn.Conv1d(embeddings_dim, embeddings_dim, kernel_size, stride=1, padding="same")


        self.softmax = nn.Softmax(dim=-1)

        self.dropout = nn.Dropout(conv_dropout)


        


    def forward(self, x: torch.Tensor, mask, **kwargs) -> torch.Tensor:
        """
        Args:
            x: [batch_size, embeddings_dim, sequence_length] embedding tensor that should be classified
            mask: [batch_size, sequence_length] mask corresponding to the zero padding used for the shorter sequecnes in the batch. All values corresponding to padding are False and the rest is True.

        Returns:
            classification: [batch_size,output_dim] tensor with logits
        """
        x = x.permute(0, 2, 1)  # [batch_size, embeddings_dim, sequence_length]

        o = self.feature_convolution(x)  # [batch_size, embeddings_dim, sequence_length]
        o = self.dropout(o)  # [batch_gsize, embeddings_dim, sequence_length]
        attention = self.attention_convolution(x)  # [batch_size, embeddings_dim, sequence_length]

        # mask out the padding to which we do not want to pay any attention (we have the padding because the sequences have different lenghts).
        # This padding is added by the dataloader when using the padded_permuted_collate function in utils/general.py
        #attention = attention.masked_fill(mask[:, None, :] == False, -1e9)
        seq_len = attention.shape[-1]  # Get actual sequence length after Conv1D

        # Print debug info
        #print(f"Mask original shape: {mask.shape}")
    
        # Ensure mask shape matches sequence length
        mask = mask[:, :, :attention.shape[-1]]  # Trim mask to match sequence length
        #print(f"Mask shape after trim: {mask.shape}")
    
        # Final sanity check
        if mask.shape[-1] != attention.shape[-1]:
            print(f"mask.shape {mask.shape}", flush=True)
            print(f"attention.shape {attention.shape}", flush=True)
            raise Exception("Mask and sequence length mismatch!")

        attention = attention.masked_fill(mask == False, -1e9)


        # code used for extracting embeddings for UMAP visualizations
        # extraction =  torch.sum(x * self.softmax(attention), dim=-1)
        # extraction = self.id0(extraction)

        o1 = torch.sum(o * self.softmax(attention), dim=-1)  # [batchsize, embeddings_dim]
        o2, _ = torch.max(o, dim=-1)  # [batchsize, embeddings_dim]
        o = torch.cat([o1, o2], dim=-1)  # [batchsize, 2*embeddings_dim]
        return o  # [batchsize, embeddings_dim]
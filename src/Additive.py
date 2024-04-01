import torch.nn as nn
import torch

class AdditiveAttention(nn.Module):
    def __init__(self, seq_len):
        super(AdditiveAttention, self).__init__()
                
        self.W1 = nn.Linear(seq_len, seq_len)
        self.W2 = nn.Linear(seq_len, seq_len)
        self.V = nn.Linear(seq_len, 1, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):

        h1 = self.W1(x)
        
        h2 = self.W2(x)
        
        attn_scores = torch.tanh(h1 + h2)

        attn_weights = self.V(attn_scores)
        
        attn_weights = self.softmax(attn_weights)

        weighted_sum = torch.sum(x * attn_weights, dim=1)

        return weighted_sum
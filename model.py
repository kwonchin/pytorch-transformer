# This is already done in transformer.py, but I wanted to write it out myself to understand it better.
import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
    def __init__(self, vocab_size, d_model):
        super(InputEmbeddings, self).__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
    
    def forward(self, x):
        return self.embed(x) * math.sqrt(self.d_model)
    

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_length: int, dropout: float):
        super.__init__()
        self.d_model = d_model
        self.seq_length = seq_length
        self.dropout = nn.Dropout(dropout)

        # Create a matrix of shape (seq_length, d_model) containing values from 0 to seq_length
        pe = torch.zeros(seq_length, d_model)

        # Create a matrix of shape (seq_length, 1) containing values from 0 to seq_length
        position = torch.arange(0, seq_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        # Apply the sin to even indices in the array; 2i
        pe[:, 0::2] = torch.sin(position * div_term)

        # Apply the cos to odd indices in the array; 2i+1
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # Add a batch dimension to the tensor

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)

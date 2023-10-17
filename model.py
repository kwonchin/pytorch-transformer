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


class LayerNormalisation(nn.Module):
    def __init__(self, eps: float = 10**-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))  # Multiplied
        self.bias = nn.Parameter(torch.zeros(1))  # Added

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class FeedForwardBlock(nn.module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)  # W1 and B1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)  # W2 and B2

    def forward(self, x):
        # (Batch, Seq, d_model) -> (Batch, Seq, d_ff) -> (Batch, Seq, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model, h, dropout):
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model must be divisible by h"

        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout):
        d_k = query.shape[-1]

        # (Batch, h, Seq, d_k) @ (Batch, h, d_k, Seq) -> (Batch, h, Seq, Seq)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)

        # Apply the mask to the attention scores
        # We replace values in the (Seq, Seq) matrix for relationships that we want to 'ignore'
        # with a very large negative number. This will cause the softmax to make the attention
        # score for these relationships to be very close to 0.
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float("-1e20"))

        attention_scores = torch.softmax(attention_scores, dim=-1)  # (Batch, h, Seq, Seq)

        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return attention_scores @ value, attention_scores

    def forward(self, q, v, k, mask):
        query = self.w_q(q)  # (Batch, Seq, d_model) -(linear transform)-> (Batch, Seq, d_model)
        key = self.w_k(k)  # (Batch, Seq, d_model) -(linear transform)-> (Batch, Seq, d_model)
        value = self.w_v(v)  # (Batch, Seq, d_model) -(linear transform)-> (Batch, Seq, d_model)

        # Split the query, key, and value into h heads
        # (Batch, Seq, d_model) -(split)-> (Batch, Seq, h, d_k) -(transpose)-> (Batch, h, Seq, d_k)
        query = query.view(query.shape[0], query.shape[1]. self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1]. self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1]. self.h, self.d_k).transpose(1, 2)

        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        # (Batch, h, Seq, d_k) --> (Batch, Seq, h, d_k) --> (Batch, Seq, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # (Batch, Seq, d_model) -(linear transform)-> (Batch, Seq, d_model)
        return self.w_o(x)


class ResidualConnection(nn.module):
    def __init__(self, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalisation()

    def forward(self, x, sublayer):
        # Technically, the paper applies sublayer(x) before dropout and normalization,
        # but we'll do it like this
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x, mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x


class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalisation()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    def forward(self, encoder_output, src_mask, trg_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, trg_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x


class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalisation()

    def forward(self, x, encoder_output, src_mask, trg_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, trg_mask)
        return self.norm(x)


class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # (Batch, Seq, d_model) -(linear transform)-> (Batch, Seq, vocab_size)
        return self.linear(x)


class TransformerBlock(nn.Module):
    def __init__(
            self,
            encoder: Encoder,
            decoder: Decoder,
            src_embedding: InputEmbeddings,
            tgt_embedding: InputEmbeddings,
            src_positional_encoding: PositionalEncoding,
            tgt_positional_encoding: PositionalEncoding,
            projection_layer: ProjectionLayer
        ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embedding = src_embedding
        self.tgt_embedding = tgt_embedding
        self.src_positional_encoding = src_positional_encoding
        self.tgt_positional_encoding = tgt_positional_encoding
        self.projection_layer = projection_layer


    def encode(self, src, src_mask):
        src = self.src_embedding(src)
        src = self.src_positional_encoding(src)
        return self.encoder(src, src_mask)
    
    def decode(self, tgt, encoder_output, src_mask, tgt_mask):
        tgt = self.tgt_embedding(tgt)
        tgt = self.tgt_positional_encoding(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self, x):
        return self.projection_layer(x)


def build_transformer(
        src_vocab_size: int,
        tgt_vocab_size: int,
        src_seq_length: int,
        tgt_seq_length: int,
        d_model: int = 512,
        h: int = 8,
        num_layers: int = 6,  # This is called N in the paper
        dropout: float = 0.1,
        d_ff: int = 2048,
    ):
    # Create the embedding layers
    src_embedding = InputEmbeddings(src_vocab_size, d_model)
    tgt_embedding = InputEmbeddings(tgt_vocab_size, d_model)
    # Create the positional encoding layers
    src_positional_encoding = PositionalEncoding(d_model, src_seq_length, dropout)
    tgt_positional_encoding = PositionalEncoding(d_model, tgt_seq_length, dropout)
    # Create the encoder and decoder blocks
    encoder = Encoder(
        nn.ModuleList(
            [
                EncoderBlock(
                    MultiHeadAttentionBlock(d_model, h, dropout),
                    FeedForwardBlock(d_model, d_ff, dropout),
                    dropout
                )
                for _ in range(num_layers)
            ]
        )
    )
    decoder = Decoder(
        nn.ModuleList(
            [
                DecoderBlock(
                    MultiHeadAttentionBlock(d_model, h, dropout),
                    MultiHeadAttentionBlock(d_model, h, dropout),
                    FeedForwardBlock(d_model, d_ff, dropout),
                    dropout
                )
                for _ in range(num_layers)
            ]
        )
    )
    # Create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # Create the transformer
    transformer = TransformerBlock(
        encoder,
        decoder,
        src_embedding,
        tgt_embedding,
        src_positional_encoding,
        tgt_positional_encoding,
        projection_layer
    )

    # Initialise the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer

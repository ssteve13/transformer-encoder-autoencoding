import torch
import torch.nn as nn
from attention import ScaledDotProductAttention
from positional_encoding import PositionalEncoding


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dim_ff, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()

        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention()

        self.fc1 = nn.Linear(d_model, dim_ff)
        self.fc2 = nn.Linear(dim_ff, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.size()

        q = self.q_linear(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_linear(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        attn_output, _ = self.attention(q, k, v, mask)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)

        x = self.norm1(x + self.dropout(attn_output))

        ff_output = self.fc2(torch.relu(self.fc1(x)))
        x = self.norm2(x + self.dropout(ff_output))

        return x


class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, dim_ff, max_len=5000):
        super(TransformerEncoder, self).__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)

        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, dim_ff)
            for _ in range(num_layers)
        ])

    def forward(self, x, mask=None):
        x = self.embedding(x)
        x = self.positional_encoding(x)

        for layer in self.layers:
            x = layer(x, mask)

        return x
class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, dim_ff, num_classes):
        super(TransformerClassifier, self).__init__()

        self.encoder = TransformerEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            dim_ff=dim_ff
        )

        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        encoder_output = self.encoder(x)
        pooled = encoder_output.mean(dim=1)
        logits = self.fc(pooled)
        return logits

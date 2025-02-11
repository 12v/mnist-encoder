import math

import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, mask=False):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads
        self.mask = mask
        self.query_weights = nn.Linear(d_model, d_model)  # Query
        self.key_weights = nn.Linear(d_model, d_model)  # Key
        self.value_weights = nn.Linear(d_model, d_model)  # Value

        self.output_layer = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, query, key, value):
        attention = torch.matmul(query, key.transpose(-2, -1))
        scaled_attention = attention / math.sqrt(self.head_dim)
        if self.mask:
            scaled_attention = scaled_attention.masked_fill(
                torch.triu(torch.ones_like(scaled_attention), diagonal=1) == 1,
                float("-inf"),
            )
        attention_activations = torch.softmax(scaled_attention, dim=-1)
        weighted_attention = torch.matmul(attention_activations, value)
        return weighted_attention

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def combine_heads(self, x):
        x = x.permute(0, 2, 1, 3)
        x = x.contiguous().view(x.size(0), -1, self.d_model)
        return x

    def forward(self, x):
        batch_size = x.size(0)

        query = self.query_weights(x)
        key = self.key_weights(x)
        value = self.value_weights(x)

        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        weighted_attention = self.scaled_dot_product_attention(query, key, value)
        combined_attention = self.combine_heads(weighted_attention)
        return self.output_layer(combined_attention)

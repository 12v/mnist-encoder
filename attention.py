import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, attention_depth, d_own, d_other, mask=False):
        super().__init__()
        self.mask = mask
        self.query_weights = nn.Linear(d_own, attention_depth)  # Query
        self.key_weights = nn.Linear(d_other, attention_depth)  # Key
        self.value_weights = nn.Linear(d_other, attention_depth)  # Value

        self.output_layer = nn.Linear(attention_depth, d_own)

    def calculate_attention(self, query, key, value):
        attention = torch.matmul(query, key.transpose(-2, -1))
        if self.mask:
            attention = attention.masked_fill(
                torch.triu(torch.ones_like(attention), diagonal=1) == 1,
                float("-inf"),
            )
        activations = torch.softmax(attention, dim=-1)
        weighted_attention = torch.matmul(activations, value)
        return weighted_attention

    def forward(self, itself, other):
        query = self.query_weights(itself)
        key = self.key_weights(other)
        value = self.value_weights(other)

        attention = self.calculate_attention(query, key, value)
        return self.output_layer(attention)

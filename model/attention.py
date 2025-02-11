import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, attention_depth, d_own, d_other, mask=False):
        super().__init__()
        self.mask = mask
        self.attention_depth = attention_depth
        self.query_weights = nn.Linear(d_own, attention_depth)  # Query
        self.key_weights = nn.Linear(d_other, attention_depth)  # Key
        self.value_weights = nn.Linear(d_other, attention_depth)  # Value

        self.output_layer = nn.Linear(attention_depth, d_own)

    def calculate_attention(self, query, key, value):
        attention = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(self.attention_depth)
        )
        if self.mask:
            attention = attention.masked_fill(
                torch.triu(torch.ones_like(attention), diagonal=1) == 1,
                float("-inf"),
            )
        raw_attention = torch.softmax(attention, dim=-1)
        weighted_attention = torch.matmul(raw_attention, value)
        return weighted_attention, raw_attention

    def forward(self, itself, other):
        query = self.query_weights(itself)
        key = self.key_weights(other)
        value = self.value_weights(other)

        attention, raw_attention = self.calculate_attention(query, key, value)
        output = self.output_layer(attention)
        return output, raw_attention

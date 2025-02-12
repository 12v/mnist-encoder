import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(
        self,
        attention_depth,
        query_dim,
        key_value_dim,
        mask=False,
    ):
        super().__init__()
        self.mask = mask
        self.attention_depth = attention_depth
        self.query_weights = nn.Linear(query_dim, attention_depth)  # Query
        self.key_weights = nn.Linear(key_value_dim, attention_depth)  # Key
        self.value_weights = nn.Linear(key_value_dim, attention_depth)  # Value

        self.output_layer = nn.Linear(attention_depth, query_dim)

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

    def forward(self, query_states, key_value_states):
        query = self.query_weights(query_states)
        key = self.key_weights(key_value_states)
        value = self.value_weights(key_value_states)

        attention, raw_attention = self.calculate_attention(query, key, value)
        output = self.output_layer(attention)
        return output, raw_attention

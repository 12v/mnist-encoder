import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(
        self,
        query_dim,
        key_value_dim,
        causal_mask=False,
    ):
        super().__init__()
        self.causal_mask = causal_mask
        self.key_value_dim = key_value_dim
        self.query_weights = nn.Linear(query_dim, key_value_dim)  # Query
        self.key_weights = nn.Linear(key_value_dim, key_value_dim)  # Key
        self.value_weights = nn.Linear(key_value_dim, key_value_dim)  # Value

        self.output_layer = nn.Linear(key_value_dim, query_dim)

    def calculate_attention(self, query, key, value, key_padding_mask):
        attention = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(self.key_value_dim)
        )
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(1)
            expanded_key_padding_mask = key_padding_mask.expand(
                -1, attention.shape[1], -1
            )
            attention = attention.masked_fill(
                expanded_key_padding_mask == 0, float("-inf")
            )

        if self.causal_mask:
            attention = attention.masked_fill(
                torch.triu(torch.ones_like(attention), diagonal=1) == 1,
                float("-inf"),
            )
        raw_attention = torch.softmax(attention, dim=-1)
        weighted_attention = torch.matmul(raw_attention, value)
        return weighted_attention, raw_attention

    def forward(self, query_states, key_value_states, key_padding_mask=None):
        query = self.query_weights(query_states)
        key = self.key_weights(key_value_states)
        value = self.value_weights(key_value_states)

        attention, raw_attention = self.calculate_attention(
            query, key, value, key_padding_mask
        )
        output = self.output_layer(attention)
        return output, raw_attention

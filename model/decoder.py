import torch
import torch.nn as nn

from model.attention import Attention
from model.encoder import Encoder
from model.positional_encoder import PositionalEncoder


class DecoderLayer(nn.Module):
    def __init__(self, d_model_decoder, d_model_encoder, num_heads):
        super().__init__()
        self.masked_self_attention = Attention(
            query_dim=d_model_decoder,
            key_value_dim=d_model_decoder,
            num_heads=num_heads,
            causal_mask=True,
        )
        self.cross_attention = Attention(
            query_dim=d_model_decoder,
            key_value_dim=d_model_encoder,
            num_heads=num_heads,
        )
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model_decoder, d_model_decoder),
            nn.ReLU(),
            nn.Linear(d_model_decoder, d_model_decoder),
        )
        self.norm1 = nn.LayerNorm(d_model_decoder)
        self.norm2 = nn.LayerNorm(d_model_decoder)
        self.norm3 = nn.LayerNorm(d_model_decoder)

    def forward(self, label_embeddings, image_encodings, padding_mask):
        x, self_attention = self.masked_self_attention(
            label_embeddings, label_embeddings, padding_mask
        )
        x = self.norm1(label_embeddings + x)
        x1, cross_attention = self.cross_attention(x, image_encodings)
        x = self.norm2(x + x1)
        x = self.norm3(x + self.feed_forward(x))
        return x, self_attention, cross_attention


class Decoder(nn.Module):
    def __init__(
        self,
        d_model_encoder,
        d_model_decoder,
        encoder_embedding_dim,
        encoder_length,
        decoder_length,
        num_encoder_layers,
        num_decoder_layers,
        vocab_size,
        num_heads,
    ):
        super().__init__()
        self.embedder = nn.Embedding(vocab_size, d_model_decoder)
        self.positional_encoder = PositionalEncoder(d_model_decoder, decoder_length)
        self.encoder = Encoder(
            d_model_encoder,
            encoder_embedding_dim,
            encoder_length,
            num_encoder_layers,
            num_heads,
        )
        self.decoder_layers = nn.ModuleList(
            [
                DecoderLayer(d_model_decoder, d_model_encoder, num_heads)
                for _ in range(num_decoder_layers)
            ]
        )
        self.output_layer = nn.Linear(d_model_decoder, vocab_size)

    def compute_loss(self, patches, input_labels, output_labels, padding_mask):
        x, _, _, _ = self.forward(patches, input_labels, padding_mask)
        x = torch.permute(x, (0, 2, 1))

        return nn.CrossEntropyLoss()(x, output_labels)

    def forward(self, patches, input_labels, padding_mask):
        image_encodings, encoder_self_attention = self.encoder(patches)

        label_embeddings = self.embedder(input_labels)
        label_embeddings = self.positional_encoder(label_embeddings)

        for layer in self.decoder_layers:
            label_embeddings, decoder_self_attention, decoder_cross_attention = layer(
                label_embeddings, image_encodings, padding_mask
            )

        x = self.output_layer(label_embeddings)

        return (
            x,
            encoder_self_attention,
            decoder_self_attention,
            decoder_cross_attention,
        )

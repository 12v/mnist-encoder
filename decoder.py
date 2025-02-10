import torch
import torch.nn as nn

from attention import Attention
from data import vocab
from encoder import Encoder
from positional_encoder import PositionalEncoder


class DecoderLayer(nn.Module):
    def __init__(self, attention_depth, d_model_decoder, d_model_encoder):
        super().__init__()
        self.masked_self_attention = Attention(
            attention_depth=attention_depth,
            d_own=d_model_decoder,
            d_other=d_model_decoder,
            mask=True,
        )
        self.cross_attention = Attention(
            attention_depth=attention_depth,
            d_own=d_model_decoder,
            d_other=d_model_encoder,
        )
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model_decoder, d_model_decoder),
            nn.ReLU(),
            nn.Linear(d_model_decoder, d_model_decoder),
        )
        self.norm1 = nn.LayerNorm(d_model_decoder)
        self.norm2 = nn.LayerNorm(d_model_decoder)
        self.norm3 = nn.LayerNorm(d_model_decoder)

    def forward(self, x, image_encodings):
        x = self.norm1(x + self.masked_self_attention(x, x))
        x = self.norm2(x + self.cross_attention(x, image_encodings))
        x = self.norm3(x + self.feed_forward(x))
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        attention_depth,
        d_model_encoder,
        d_model_decoder,
        encoder_embedding_dim,
        decoder_embedding_dim,
        encoder_length,
        decoder_length,
    ):
        super().__init__()
        self.embedder = nn.Embedding(decoder_embedding_dim, d_model_decoder)
        self.positional_encoder = PositionalEncoder(d_model_decoder, decoder_length)
        self.encoder = Encoder(
            attention_depth, d_model_encoder, encoder_embedding_dim, encoder_length
        )
        self.decoder_layers = nn.ModuleList(
            [
                DecoderLayer(attention_depth, d_model_decoder, d_model_encoder)
                for _ in range(6)
            ]
        )
        self.output_layer = nn.Linear(d_model_decoder, len(vocab))
        self.softmax = nn.Softmax(dim=-1)

    def compute_loss(self, patches, input_labels, output_labels):
        x = self.forward(patches, input_labels)
        x = torch.permute(x, (0, 2, 1))

        return nn.CrossEntropyLoss()(x, output_labels)

    def forward(self, patches, input_labels):
        image_encodings = self.encoder(patches)

        label_embeddings = self.embedder(input_labels)
        label_embeddings = self.positional_encoder(label_embeddings)

        for layer in self.decoder_layers:
            label_embeddings = layer(label_embeddings, image_encodings)

        x = self.output_layer(label_embeddings)
        x = self.softmax(x)
        # print(x.shape)
        # print(x)
        # exit()

        return x

import torch
import torch.nn as nn

from model.encoder import Encoder


class MaskedAutoencoder(nn.Module):
    def __init__(
        self,
        d_model_encoder,
        encoder_embedding_dim,
        encoder_length,
        num_encoder_layers,
        num_heads,
    ):
        super().__init__()
        self.encoder = Encoder(
            d_model_encoder,
            encoder_embedding_dim,
            encoder_length,
            num_encoder_layers,
            num_heads,
        )
        self.decoder = nn.Sequential(
            nn.Linear(d_model_encoder, 1024),
            nn.ReLU(),
            nn.Linear(1024, encoder_embedding_dim),
        )

    def compute_loss(self, original):
        reconstructed = self(original)
        return nn.MSELoss()(reconstructed, original)

    def forward(self, x):
        mask = torch.rand(x.shape) > 0.3  # 30% masking
        mask = mask.to(x.device)
        x_masked = x * mask
        features, _ = self.encoder(x_masked)
        reconstructed = self.decoder(features)
        return reconstructed

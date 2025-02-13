import multiprocessing as mp
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.flickr30k import (
    Flickr30kDataset,
    height,
    test_ds,
    train_ds,
    vocab_size,
    width,
)
from model.decoder import Decoder
from model.utils import device
from params_flickr import (
    attention_depth,
    d_model_decoder,
    d_model_encoder,
    decoder_length,
    num_decoder_layers,
    num_encoder_layers,
    patch_dim,
)

num_workers = 6 if torch.cuda.is_available() else 0
persistent_workers = True if num_workers > 0 else False
batch_size = 400 if torch.cuda.is_available() else 200
initial_lr = 0.03 if torch.cuda.is_available() else 0.01


def train():
    num_epochs = 40

    torch.manual_seed(7)

    val_dataloader = DataLoader(
        Flickr30kDataset(test_ds, patch_dim),
        batch_size=batch_size,
        drop_last=True,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
    )

    train_dataloader = DataLoader(
        Flickr30kDataset(train_ds, patch_dim),
        batch_size=batch_size,
        drop_last=True,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
    )

    model = Decoder(
        attention_depth=attention_depth,
        # internal dimensions
        d_model_encoder=d_model_encoder,
        d_model_decoder=d_model_decoder,
        encoder_embedding_dim=width * height // patch_dim // patch_dim,
        # length of the input and output sequences
        encoder_length=patch_dim * patch_dim,
        decoder_length=decoder_length,
        vocab_size=vocab_size,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
    )

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)

    # Get datasets and dataloaders
    print("Loading training data...")

    # Training loop
    for epoch in range(num_epochs):
        batch_losses = []

        train_loop = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch + 1}/{num_epochs}",
            total=len(train_ds) * 5 // batch_size,
        )

        for (
            image_batch,
            input_label_batch,
            output_label_batch,
            padding_mask,
        ) in train_loop:
            model.train()
            optimizer.zero_grad()

            loss = model.compute_loss(
                image_batch.to(device),
                input_label_batch.to(device),
                output_label_batch.to(device),
                padding_mask.to(device),
            )

            loss.backward()
            optimizer.step()

            batch_losses.append(loss.item())
            train_loop.set_postfix(loss=f"{sum(batch_losses) / len(batch_losses):.4f}")

        val_losses = []
        val_loop = tqdm(
            val_dataloader,
            desc=f"Validation for epoch {epoch + 1}/{num_epochs}",
            total=len(test_ds) * 5 // batch_size,
        )

        for (
            image_batch,
            input_label_batch,
            output_label_batch,
            padding_mask,
        ) in val_loop:
            model.eval()
            with torch.no_grad():
                loss = model.compute_loss(
                    image_batch.to(device),
                    input_label_batch.to(device),
                    output_label_batch.to(device),
                    padding_mask.to(device),
                )
                val_losses.append(loss.item())

        os.makedirs("weights", exist_ok=True)
        torch.save(model.state_dict(), f"weights/model_flickr_{epoch}.pth")
        print(f"Validation loss: {sum(val_losses) / len(val_losses):.4f}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    train()

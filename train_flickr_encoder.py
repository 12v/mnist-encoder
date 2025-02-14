import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.imagenet import (
    ImageNetEncoderDataset,
    height,
    width,
)
from model.masked_autoencoder import MaskedAutoencoder
from model.utils import device
from params_flickr import (
    d_model_encoder,
    num_encoder_layers,
    num_heads,
    patch_dim,
)

num_workers = 1 if torch.cuda.is_available() else 2
batch_size = 400 if torch.cuda.is_available() else 100
initial_lr = 0.01 if torch.cuda.is_available() else 0.01
# if only to prevent the data being reloaded from HF and timing out :(
persistent_workers = True if torch.cuda.is_available() else False


def train():
    num_epochs = 40

    torch.manual_seed(7)

    # val_dataloader = DataLoader(
    #     Flickr30kEncoderDataset(test_ds, patch_dim),
    #     batch_size=batch_size,
    #     drop_last=True,
    #     num_workers=num_workers,
    #     persistent_workers=persistent_workers,
    # )

    # train_dataloader = DataLoader(
    #     Flickr30kEncoderDataset(train_ds, patch_dim),
    #     batch_size=batch_size,
    #     drop_last=True,
    #     num_workers=num_workers,
    #     persistent_workers=persistent_workers,
    # )

    # val_dataloader = DataLoader(
    #     ImageNetEncoderDataset(val_data, patch_dim),
    #     batch_size=batch_size,
    #     drop_last=True,
    #     num_workers=num_workers,
    #     persistent_workers=persistent_workers,
    # )

    train_dataloader = DataLoader(
        ImageNetEncoderDataset(patch_dim),
        batch_size=batch_size,
        drop_last=True,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
    )

    model = MaskedAutoencoder(
        d_model_encoder=d_model_encoder,
        encoder_embedding_dim=width * height * 3 // patch_dim // patch_dim,
        encoder_length=patch_dim * patch_dim,
        num_encoder_layers=num_encoder_layers,
        num_heads=num_heads,
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
            # total=len(train_data) // batch_size,
        )

        for i, image_batch in enumerate(train_loop):
            model.train()
            optimizer.zero_grad()

            loss = model.compute_loss(
                image_batch.to(device),
            )

            loss.backward()
            optimizer.step()

            batch_losses.append(loss.item())
            train_loop.set_postfix(loss=f"{sum(batch_losses) / len(batch_losses):.4f}")

            if i > 50:
                break

        # val_losses = []
        # val_loop = tqdm(
        #     val_dataloader,
        #     desc=f"Validation for epoch {epoch + 1}/{num_epochs}",
        #     total=len(val_data) // batch_size,
        # )

        # for image_batch in val_loop:
        #     model.eval()
        #     with torch.no_grad():
        #         loss = model.compute_loss(image_batch.to(device))
        #         val_losses.append(loss.item())

        os.makedirs("weights", exist_ok=True)
        torch.save(model.state_dict(), f"weights/model_flickr_encoder_{epoch}.pth")
        # print(f"Validation loss: {sum(val_losses) / len(val_losses):.4f}")


if __name__ == "__main__":
    # mp.set_start_method("spawn", force=True)
    train()

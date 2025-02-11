import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import Dataset
from decoder import Decoder
from mnist import train_data
from utils import device


def train():
    num_epochs = 20

    torch.manual_seed(7)

    batch_size = 500

    train_dataloader = DataLoader(
        Dataset(train_data),
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
    )

    model = Decoder(
        attention_depth=24,
        # internal dimensions
        d_model_encoder=64,
        d_model_decoder=12,
        # embedding dimensions
        encoder_embedding_dim=196,
        decoder_embedding_dim=13,
        # length of the input and output sequences
        encoder_length=16,
        decoder_length=5,
    )

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Get datasets and dataloaders
    print("Loading training data...")

    # Training loop
    for epoch in range(num_epochs):
        batch_losses = []

        train_loop = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch + 1}/{num_epochs}",
            total=len(train_dataloader),
        )

        for image_batch, input_label_batch, output_label_batch in train_loop:
            model.train()
            optimizer.zero_grad()

            loss = model.compute_loss(
                image_batch.to(device),
                input_label_batch.to(device),
                output_label_batch.to(device),
            )

            loss.backward()
            optimizer.step()

            batch_losses.append(loss.item())
            train_loop.set_postfix(loss=f"{sum(batch_losses) / len(batch_losses):.4f}")

    # save the model weights
    torch.save(model.state_dict(), "model.pth")


if __name__ == "__main__":
    train()

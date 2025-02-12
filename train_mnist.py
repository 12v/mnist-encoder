import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.mnist import MnistDataset, test_data, train_data, vocab
from model.decoder import Decoder
from model.utils import device
from params_mnist import (
    attention_depth,
    d_model_decoder,
    d_model_encoder,
    decoder_length,
    encoder_embedding_dim,
    encoder_length,
    num_decoder_layers,
    num_encoder_layers,
    patch_dim,
)


def train():
    num_epochs = 40

    torch.manual_seed(7)

    batch_size = 500

    val_dataloader = DataLoader(
        MnistDataset(test_data, patch_dim),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    model = Decoder(
        attention_depth=attention_depth,
        # internal dimensions
        d_model_encoder=d_model_encoder,
        d_model_decoder=d_model_decoder,
        # embedding dimensions
        encoder_embedding_dim=encoder_embedding_dim,
        # length of the input and output sequences
        encoder_length=encoder_length,
        decoder_length=decoder_length,
        vocab_size=len(vocab),
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
    )

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Get datasets and dataloaders
    print("Loading training data...")

    # Training loop
    for epoch in range(num_epochs):
        train_dataloader = DataLoader(
            MnistDataset(train_data, patch_dim),
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
        )

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

        val_losses = []
        for image_batch, input_label_batch, output_label_batch in val_dataloader:
            model.eval()
            with torch.no_grad():
                loss = model.compute_loss(
                    image_batch.to(device),
                    input_label_batch.to(device),
                    output_label_batch.to(device),
                )
                val_losses.append(loss.item())

        torch.save(model.state_dict(), f"weights/model_{epoch}.pth")
        print(f"Validation loss: {sum(val_losses) / len(val_losses):.4f}")


if __name__ == "__main__":
    train()

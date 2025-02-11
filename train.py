import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import Dataset
from data.mnist import test_data, train_data
from data.tokenizer import vocab
from model.decoder import Decoder
from model.utils import device

patch_dim = 4

attention_depth = 24
d_model_encoder = 64
d_model_decoder = 12

encoder_embedding_dim = 196
decoder_embedding_dim = 13

encoder_length = patch_dim * patch_dim
decoder_length = 5

num_encoder_layers = 6
num_decoder_layers = 6


def train():
    num_epochs = 20

    torch.manual_seed(7)

    batch_size = 500

    train_dataloader = DataLoader(
        Dataset(train_data, patch_dim),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    val_dataloader = DataLoader(
        Dataset(test_data, patch_dim),
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
        decoder_embedding_dim=decoder_embedding_dim,
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

        print(f"Validation loss: {sum(val_losses) / len(val_losses):.4f}")

    # save the model weights
    torch.save(model.state_dict(), "weights/model.pth")


if __name__ == "__main__":
    train()

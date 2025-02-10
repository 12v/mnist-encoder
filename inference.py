import torch

from data import (
    create_patches,
    generate_grids,
    tokenize,
    train_data,
    visualize_grid,
    vocab,
)
from decoder import Decoder

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

model.load_state_dict(torch.load("model.pth"))

model.eval()
with torch.no_grad():
    for grid, _ in generate_grids(train_data):
        patches = create_patches(grid)

        sequence = ["<start>"]

        for i in range(4):
            tokens = torch.stack((tokenize(sequence),))
            print("tokens", tokens.shape)
            output = model(patches, tokens)
            sequence.append(vocab[torch.argmax(output[0], dim=1)[i].item()])

        print(sequence[1:])
        visualize_grid(grid)

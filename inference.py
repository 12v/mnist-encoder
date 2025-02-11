import torch

from data.images import create_patches, flatten_patches
from data.mnist import get_images_and_labels, test_data
from data.tokenizer import tokenize, vocab
from data.visualization import visualize_image
from model.decoder import Decoder

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
    vocab_size=len(vocab),
    num_encoder_layers=6,
    num_decoder_layers=6,
)

model.load_state_dict(torch.load("weights/model.pth"))

model.eval()
with torch.no_grad():
    images, _ = get_images_and_labels(test_data)
    for image in images:
        patches = create_patches(image)
        patches = flatten_patches(patches)

        sequence = ["<start>"]

        for i in range(4):
            tokens = torch.stack((tokenize(sequence),))
            print("tokens", tokens.shape)
            output = model(patches, tokens)
            sequence.append(vocab[torch.argmax(output[0], dim=1)[i].item()])

        print(sequence[1:])
        visualize_image(image)

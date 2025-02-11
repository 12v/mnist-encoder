import torch

from data.images import create_patches, flatten_patches
from data.mnist import get_images_and_labels, test_data
from data.tokenizer import tokenize, vocab
from data.visualization import visualize_attention
from model.decoder import Decoder
from train import (
    attention_depth,
    d_model_decoder,
    d_model_encoder,
    decoder_embedding_dim,
    decoder_length,
    encoder_embedding_dim,
    encoder_length,
    num_decoder_layers,
    num_encoder_layers,
    patch_dim,
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

model.load_state_dict(torch.load("weights/model.pth"))

model.eval()
with torch.no_grad():
    images, _ = get_images_and_labels(test_data)
    for image in images:
        patches = create_patches(image, patch_dim)
        patches = flatten_patches(patches, patch_dim)

        input_sequence = ["<start>"]
        output_sequence = []

        encoder_self_attention = None
        decode_self_attention = None
        decode_cross_attention = None

        for i in range(decoder_length):
            tokens = torch.stack((tokenize(input_sequence),))
            (
                output,
                encoder_self_attention,
                decode_self_attention,
                decode_cross_attention,
            ) = model(patches, tokens)

            output_token = torch.argmax(output[0], dim=1)[i].item()
            output_text = vocab[output_token]
            input_sequence.append(output_text)
            output_sequence.append(output_text)

            if i == decoder_length - 1:
                encoder_self_attention = encoder_self_attention
                decode_self_attention = decode_self_attention
                decode_cross_attention = decode_cross_attention

        print(output_sequence)
        visualize_attention(
            image, encoder_self_attention, decode_self_attention, decode_cross_attention
        )

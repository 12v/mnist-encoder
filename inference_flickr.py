import torch

from data.flickr30k import (
    height,
    image_and_caption_generator,
    start_token,
    test_ds,
    tokenizer,
    vocab_size,
    width,
)
from data.images import create_patches, flatten_patches
from data.visualization import visualize_image
from model.decoder import Decoder
from model.utils import device
from params_flickr import (
    d_model_decoder,
    d_model_encoder,
    decoder_length,
    num_decoder_layers,
    num_encoder_layers,
    num_heads,
    patch_dim,
)

model = Decoder(
    # internal dimensions
    d_model_encoder=d_model_encoder,
    d_model_decoder=d_model_decoder,
    encoder_embedding_dim=width * height * 3 // patch_dim // patch_dim,
    # length of the input and output sequences
    encoder_length=patch_dim * patch_dim,
    decoder_length=decoder_length,
    vocab_size=vocab_size,
    num_encoder_layers=num_encoder_layers,
    num_decoder_layers=num_decoder_layers,
    num_heads=num_heads,
)


model.load_state_dict(torch.load("weights/model_flickr_0_gpu.pth", map_location=device))

model.to(device)

model.eval()
with torch.no_grad():
    for image, caption in image_and_caption_generator(test_ds):
        patches = create_patches(image, patch_dim)
        patches = flatten_patches(patches, patch_dim)

        input_tokens = [tokenizer.get_id_for_token(start_token)]
        output_tokens = []

        encoder_self_attention = None
        decode_self_attention = None
        decode_cross_attention = None

        for i in range(decoder_length):
            tokens = torch.stack((torch.tensor(input_tokens),))
            (
                output,
                encoder_self_attention,
                decode_self_attention,
                decode_cross_attention,
            ) = model(
                patches.unsqueeze(0).to(device),
                tokens.to(device),
                torch.ones_like(tokens).to(device),
            )

            print(output[0][0][1:])
            print(torch.argmax(output[0][0][1:], dim=0))
            output_token = torch.argmax(output[0][0][1:], dim=0).item() + 1
            input_tokens.append(output_token)
            output_tokens.append(output_token)

            print(output_tokens)
            output_text = tokenizer.decode(output_tokens)

            if i == decoder_length - 1:
                encoder_self_attention = encoder_self_attention
                decode_self_attention = decode_self_attention
                decode_cross_attention = decode_cross_attention

        print(output_text)
        visualize_image(image)
        exit()
        # visualize_attention(
        #     image,
        #     encoder_self_attention,
        #     decode_self_attention,
        #     decode_cross_attention,
        #     output_text,
        # )
        visualize_image(image)

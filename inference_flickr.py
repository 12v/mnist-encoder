import torch
import torch.nn.functional as F

from data.flickr30k import (
    finish_token,
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


model.load_state_dict(
    torch.load("weights/model_flickr_0_gpu_vocab2k.pth", map_location=device)
)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total Parameters: {total_params}")

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

            softmax_output = F.softmax(output[0][0][1:], dim=-1)
            output_token = torch.multinomial(softmax_output, 1).item()

            input_tokens.append(output_token)
            output_tokens.append(output_token)

            print(output_tokens, end="\r")
            output_text = tokenizer.decode(output_tokens)

            encoder_self_attention = encoder_self_attention
            decode_self_attention = decode_self_attention
            decode_cross_attention = decode_cross_attention

            if output_token == tokenizer.get_id_for_token(finish_token):
                break

        print("\n")
        print(output_text)
        visualize_image(image)
        # visualize_attention(
        #     image,
        #     encoder_self_attention,
        #     decode_self_attention,
        #     decode_cross_attention,
        #     output_text,
        # )

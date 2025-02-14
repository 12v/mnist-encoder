import io

import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from PIL import Image

from data.flickr30k_inference import (
    finish_token,
    height,
    process_image,
    start_token,
    tokenizer,
    vocab_size,
    width,
)
from data.images import create_patches, flatten_patches
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

app = FastAPI()

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

total_params = sum(p.numel() for p in model.parameters())
print(f"Total Parameters: {total_params}")

model.to(device)
model.eval()


async def processing_image(file: UploadFile):
    with torch.no_grad():
        image = process_image(file)
        patches = create_patches(image, patch_dim)
        patches = flatten_patches(patches, patch_dim)

        input_tokens = [tokenizer.get_id_for_token(start_token)]
        output_tokens = []

        output_string = ""

        for i in range(decoder_length):
            tokens = torch.stack((torch.tensor(input_tokens),))
            (
                output,
                _,
                _,
                _,
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

            new_output_string = tokenizer.decode(output_tokens)

            yield new_output_string[len(output_string) :]

            output_string = new_output_string

            if output_token == tokenizer.get_id_for_token(finish_token):
                break


@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))
    return StreamingResponse(processing_image(image), media_type="text/plain")


@app.get("/")
async def root():
    # return the index.html file
    return FileResponse("index.html")

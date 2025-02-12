import os
import random

import sentencepiece as spm
import torch
from datasets import load_dataset
from PIL import Image
from torch.utils.data import IterableDataset
from torchvision.transforms import ToTensor

from data.images import create_patches, flatten_patches, normalize_and_standardize

script_dir = os.path.dirname(os.path.abspath(__file__))

ds = load_dataset("nlphuji/flickr30k", split="test", trust_remote_code=True)

length = len(ds)
train_length = int(length * 0.8)
test_length = length - train_length

train_ds = ds.select(range(train_length))
test_ds = ds.select(range(train_length, length))

width = 500
height = 500

start_token = "<s>"
finish_token = "</s>"

vocab_size = 10000


class Flickr30kTokenizer:
    def __init__(self, ds):
        corpus_path = os.path.join(script_dir, "corpus.txt")
        model_path = os.path.join(script_dir, "flickr30k.model")

        if not os.path.exists(model_path):
            if not os.path.exists(corpus_path):
                self.create_corpus(ds, corpus_path)

            self.train_model(corpus_path, model_path)

        self.sp = spm.SentencePieceProcessor(model_file=model_path)

    def create_corpus(self, ds, path):
        with open(path, "w") as f:
            for captions in ds["caption"]:
                for caption in captions:
                    f.write(caption + "\n")

    def train_model(self, corpus_path, model_path):
        with open(model_path, "wb") as f:
            spm.SentencePieceTrainer.train(
                input=corpus_path,
                model_writer=f,
                vocab_size=vocab_size,
            )

    def decode(self, tokens):
        return self.sp.decode(tokens)

    def encode(self, caption, length=50):
        tokens = self.sp.encode(caption)
        tokens = tokens[:length]
        tokens = tokens + [self.sp.piece_to_id("<unk>")] * (length - len(tokens))
        return [self.sp.piece_to_id(start_token)] + tokens, tokens + [
            self.sp.piece_to_id(finish_token)
        ]

    def get_id_for_token(self, token):
        return self.sp.piece_to_id(token)


tokenizer = Flickr30kTokenizer(ds)


def pad_photo(photo):
    if photo.size[0] == width and photo.size[1] == height:
        return photo

    else:
        original_width, original_height = photo.size
        new_image = Image.new("RGB", (width, height), (0, 0, 0))
        new_image.paste(
            photo,
            (
                (width - original_width) // 2,
                (height - original_height) // 2,
            ),
        )
        return new_image


def image_and_caption_generator(ds):
    while True:
        photo_index = random.randint(0, len(ds) - 1)
        photo = ds[photo_index]["image"]
        captions = ds[photo_index]["caption"]
        caption = random.choice(captions)

        photo = pad_photo(photo)
        photo = photo.split()[0]
        tensor = ToTensor()(photo)
        tensor = normalize_and_standardize(tensor.squeeze(0))

        yield tensor, caption


def prep_for_training(tensor, caption, patch_dim):
    patches = create_patches(tensor, patch_dim)
    patches = flatten_patches(patches, patch_dim)
    input_caption, output_caption = tokenizer.encode(caption)

    return patches, torch.tensor(input_caption), torch.tensor(output_caption)


class Flickr30kDataset(IterableDataset):
    def __init__(self, ds, patch_dim):
        self.ds = ds
        self.patch_dim = patch_dim

    def __iter__(self):
        while True:
            photo, caption = next(image_and_caption_generator(self.ds))
            yield prep_for_training(photo, caption, self.patch_dim)


if __name__ == "__main__":
    tensor, input_caption, output_caption = next(image_and_caption_generator(train_ds))
    print("input_caption", input_caption)
    print("input_caption_decoded", tokenizer.decode(input_caption))
    print("output_caption", output_caption)
    print("output_caption_decoded", tokenizer.decode(output_caption))

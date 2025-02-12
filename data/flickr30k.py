import os
import pickle
import random

import sentencepiece as spm
import torch
from datasets import load_dataset
from PIL import Image
from torch.utils.data import IterableDataset
from torchvision.transforms import ToTensor

from data.images import create_patches, flatten_patches, normalize_and_standardize

script_dir = os.path.dirname(os.path.abspath(__file__))
pickle_path = os.path.join(script_dir, "flickr30k.pkl.gz")

width = 500
height = 500


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


def prep_patches(photo, patch_dim):
    photo = pad_photo(photo)
    photo = photo.split()[0]
    tensor = ToTensor()(photo)
    tensor = normalize_and_standardize(tensor.squeeze(0))
    patches = create_patches(tensor, patch_dim)
    patches = flatten_patches(patches, patch_dim)
    return patches


if not os.path.exists(pickle_path):
    ds = load_dataset("nlphuji/flickr30k", split="test", trust_remote_code=True)

    prepped_train_ds = []
    total_rows = len(ds)

    for idx, row in enumerate(ds):
        # Prepare the image patches
        image_patches = prep_patches(row["image"], 10)

        # Append to the prepared dataset
        prepped_train_ds.append({"image": image_patches, "caption": row["caption"]})

        # Log progress every 100 rows
        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}/{total_rows} rows.")

        if idx > 1000:
            break

    with open(pickle_path, "wb") as f:
        pickle.dump(prepped_train_ds, f)

    ds = prepped_train_ds
else:
    with open(pickle_path, "rb") as f:
        ds = pickle.load(f)


length = len(ds)
train_length = int(length * 0.8)
test_length = length - train_length

train_ds = ds[:train_length]
test_ds = ds[train_length:]

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


tokenizer = Flickr30kTokenizer(ds)


def image_and_caption_generator(ds):
    while True:
        photo_index = random.randint(0, len(ds) - 1)
        patches = ds[photo_index]["image"]
        captions = ds[photo_index]["caption"]
        caption = random.choice(captions)
        input_caption, output_caption = tokenizer.encode(caption)

        yield patches, torch.tensor(input_caption), torch.tensor(output_caption)


class Flickr30kDataset(IterableDataset):
    def __init__(self, ds):
        self.ds = ds

    def __iter__(self):
        return image_and_caption_generator(self.ds)


if __name__ == "__main__":
    tensor, input_caption, output_caption = next(image_and_caption_generator(train_ds))
    print("input_caption", input_caption)
    print("input_caption_decoded", tokenizer.decode(input_caption))
    print("output_caption", output_caption)
    print("output_caption_decoded", tokenizer.decode(output_caption))

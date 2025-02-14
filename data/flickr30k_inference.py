import os
import random

import sentencepiece as spm
import torch
from torch.utils.data import IterableDataset
from torchvision.transforms import ToTensor

from data.images import (
    create_patches,
    flatten_patches,
    normalize_and_standardize,
    pad_photo,
)

script_dir = os.path.dirname(os.path.abspath(__file__))


width = 500
height = 500

start_token = "<s>"
finish_token = "</s>"
padding_token = "<pad>"

vocab_size = 2000


class Flickr30kTokenizer:
    def __init__(self, ds=None):
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
                control_symbols=[padding_token],
            )

    def decode(self, tokens):
        return self.sp.decode(tokens)

    def encode(self, caption, length=50):
        tokens = self.sp.encode(caption)
        tokens = tokens[: length - 1]

        padding_id = self.sp.piece_to_id(padding_token)

        input_encoding = [self.sp.piece_to_id(start_token)] + tokens
        output_encoding = tokens + [self.sp.piece_to_id(finish_token)]

        padded_input_encoding = input_encoding + [padding_id] * (
            length - len(input_encoding)
        )
        padded_output_encoding = output_encoding + [padding_id] * (
            length - len(output_encoding)
        )
        padding_mask = [1] * len(input_encoding) + [0] * (length - len(input_encoding))

        return padded_input_encoding, padded_output_encoding, padding_mask

    def get_id_for_token(self, token):
        return self.sp.piece_to_id(token)


tokenizer = Flickr30kTokenizer()


def image_and_caption_generator(ds):
    options = []
    for i in range(len(ds)):
        for j in range(5):
            options.append((i, j))

    random.shuffle(options)

    for photo_index, caption_index in options:
        photo = ds[photo_index]["image"]
        caption = ds[photo_index]["caption"][caption_index]

        photo = process_image(photo)

        yield photo, caption


def image_generator(ds):
    options = []
    for i in range(len(ds)):
        options.append(i)

    random.shuffle(options)

    for photo_index in options:
        photo = ds[photo_index]["image"]

        photo = process_image(photo)

        yield photo


def process_image(image):
    photo = pad_photo(image, width, height)
    photo = ToTensor()(photo)
    photo = normalize_and_standardize(photo.squeeze(0))
    return photo


def prep_image(tensor, patch_dim):
    patches = create_patches(tensor, patch_dim)
    patches = flatten_patches(patches, patch_dim)
    return patches


def prep_caption(caption):
    input_caption, output_caption, padding_mask = tokenizer.encode(caption)
    return (
        torch.tensor(input_caption),
        torch.tensor(output_caption),
        torch.tensor(padding_mask),
    )


class Flickr30kDataset(IterableDataset):
    def __init__(self, ds, patch_dim):
        self.ds = ds
        self.patch_dim = patch_dim

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        dataset = self.ds
        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            dataset_size = len(self.ds)
            per_worker = dataset_size // num_workers
            dataset = self.ds.select(
                range(worker_id * per_worker, (worker_id + 1) * per_worker)
            )
        generator = image_and_caption_generator(dataset)
        for photo, caption in generator:
            prepped_image = prep_image(photo, self.patch_dim)
            input_caption, output_caption, padding_mask = prep_caption(caption)
            yield prepped_image, input_caption, output_caption, padding_mask


class Flickr30kEncoderDataset(IterableDataset):
    def __init__(self, ds, patch_dim):
        self.ds = ds
        self.patch_dim = patch_dim

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        dataset = self.ds
        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            dataset_size = len(self.ds)
            per_worker = dataset_size // num_workers
            dataset = self.ds.select(
                range(worker_id * per_worker, (worker_id + 1) * per_worker)
            )
        generator = image_generator(dataset)
        for photo in generator:
            yield prep_image(photo, self.patch_dim)

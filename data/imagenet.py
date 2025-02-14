import os
import random

from PIL import Image
from torch.utils.data import IterableDataset
from torchvision.transforms import ToTensor

# update with huggingface token
# login("")
# ds = load_dataset("imagenet-1k", trust_remote_code=True, split="train")
# train_ds = ds["train"]
# train_ds[0]["image"]
from data.images import create_patches, flatten_patches, normalize_and_standardize

script_dir = os.path.dirname(os.path.abspath(__file__))

imagenet_dir = os.path.join(script_dir, "..", "imagenet")

file_names = [
    os.path.abspath(os.path.join(imagenet_dir, file_name))
    for file_name in os.listdir(imagenet_dir)
]


# # Define the image transformation pipeline
# transform = transforms.Compose([
#     transforms.Resize((500, 500)),  # Resize image to the size expected by pretrained models
#     transforms.ToTensor(),          # Convert the image to a tensor
# ])

# # Load ImageNet training set (automatically downloads if not present locally)
# train_data = datasets.ImageFolder(
#     root='./data/imagenet',                  # Path to the directory to store ImageNet
#     transform=transform             # Apply the transformation pipeline
# )

# # # Load ImageNet validation set (if you also need validation data)
# # val_data = datasets.ImageFolder(
# #     root='./data/imagenet/val',
#     transform=transform
# )


height = 500
width = 500


def prep_image(tensor, patch_dim):
    patches = create_patches(tensor, patch_dim)
    patches = flatten_patches(patches, patch_dim)
    return patches


def image_generator():
    # copy file names
    file_names_copy = file_names.copy()
    random.shuffle(file_names_copy)

    for file_name in file_names_copy:
        photo = Image.open(file_name)
        photo = photo.convert("RGB")
        photo = photo.resize((width, height))

        # photo = pad_photo(photo, width, height)

        photo = ToTensor()(photo)
        photo = normalize_and_standardize(photo.squeeze(0))

        yield photo


class ImageNetEncoderDataset(IterableDataset):
    def __init__(self, patch_dim):
        self.patch_dim = patch_dim

    def __iter__(self):
        generator = image_generator()
        for photo in generator:
            yield prep_image(photo, self.patch_dim)

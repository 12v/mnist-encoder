from torch.utils.data import Dataset

from data.images import create_patches, flatten_patches
from data.mnist import get_images_and_labels
from data.tokenizer import tokenize_input_labels, tokenize_output_labels


class Dataset(Dataset):
    def __init__(self, data, patch_dim):
        super().__init__()
        images, labels = get_images_and_labels(data)
        patches = [create_patches(image, patch_dim) for image in images]
        patches = [flatten_patches(patch, patch_dim) for patch in patches]
        self.patches = patches
        self.input_labels = [tokenize_input_labels(label) for label in labels]
        self.output_labels = [tokenize_output_labels(label) for label in labels]

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, index):
        return self.patches[index], self.input_labels[index], self.output_labels[index]

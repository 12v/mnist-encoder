from torch.utils.data import Dataset

from data import create_patches, flatten_patches, get_grids
from tokenizer import tokenize_input_labels, tokenize_output_labels


class Dataset(Dataset):
    def __init__(self, data):
        super().__init__()
        grids, labels = get_grids(data)
        patches = [create_patches(grid) for grid in grids]
        patches = [flatten_patches(patch) for patch in patches]
        self.patches = patches
        self.input_labels = [tokenize_input_labels(label) for label in labels]
        self.output_labels = [tokenize_output_labels(label) for label in labels]

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, index):
        return self.patches[index], self.input_labels[index], self.output_labels[index]

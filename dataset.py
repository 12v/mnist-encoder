from torch.utils.data import Dataset

from data import generate_patches


class Dataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data
        self.generate_patches = [patches for patches in generate_patches(self.data)]

    def __len__(self):
        return len(self.generate_patches)

    def __getitem__(self, index):
        return self.generate_patches[index]

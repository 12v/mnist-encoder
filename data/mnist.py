import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

from data.images import create_patches, flatten_patches, normalize_and_standardize

train_data = datasets.MNIST(
    root="./resources", train=True, download=True, transform=transforms.ToTensor()
)

train_data.data = normalize_and_standardize(train_data.data)

test_data = datasets.MNIST(
    root="./resources", train=False, download=True, transform=transforms.ToTensor()
)

test_data.data = normalize_and_standardize(test_data.data)


def get_images_and_labels(data):
    outer_random_indices = torch.randperm(data.data.shape[0])
    images = []
    labels = []

    for i in range(0, data.data.shape[0], 4):
        if i + 4 > data.data.shape[0]:
            break

        random_indices = outer_random_indices[i : i + 4]
        random_images = data.data[random_indices]
        random_labels = data.targets[random_indices]

        left_column = torch.cat((random_images[0], random_images[2]), dim=0)
        right_column = torch.cat((random_images[1], random_images[3]), dim=0)

        img_grid = torch.cat((left_column, right_column), dim=1)

        images.append(img_grid)
        labels.append(random_labels)

    return images, labels


class MnistDataset(Dataset):
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


vocab = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "<start>",
    "<finish>",
    "<pad>",
]


def tokenize_input_labels(labels):
    labels = labels.tolist()
    sentence = ["<start>"] + [str(label) for label in labels]
    return tokenize(sentence)


def tokenize_output_labels(labels):
    labels = labels.tolist()
    sentence = [str(label) for label in labels] + ["<finish>"]
    return tokenize(sentence)


def tokenize(labels):
    return torch.tensor([vocab.index(c) for c in labels], dtype=torch.long)

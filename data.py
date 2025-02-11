import torch
from torchvision import datasets, transforms

from tokenizer import tokenize_input_labels, tokenize_output_labels
from visualization import visualize

train_data = datasets.MNIST(
    root="./data", train=True, download=True, transform=transforms.ToTensor()
)

train_data.data = train_data.data / 255

test_data = datasets.MNIST(
    root="./data", train=False, download=True, transform=transforms.ToTensor()
)

test_data.data = test_data.data / 255


def generate_grids(data):
    outer_random_indices = torch.randperm(data.data.shape[0])

    for i in range(0, data.data.shape[0], 4):
        if i + 4 > data.data.shape[0]:
            break

        random_indices = outer_random_indices[i : i + 4]
        random_images = data.data[random_indices]
        random_labels = data.targets[random_indices]

        top_row = torch.cat((random_images[0], random_images[1]), dim=0)
        bottom_row = torch.cat((random_images[2], random_images[3]), dim=0)

        img_grid = torch.cat((top_row, bottom_row), dim=1)

        yield img_grid, random_labels


def create_patches(grid):
    patch_count = 4
    patch_size = grid.shape[0] // patch_count

    grid = grid.unfold(0, patch_size, patch_size)
    grid = grid.unfold(1, patch_size, patch_size)

    patches = grid.reshape(-1, patch_size, patch_size)

    flat_patches = patches.reshape(16, -1)

    return flat_patches


def generate_patches(data):
    outer_random_indices = torch.randperm(data.data.shape[0])

    for i in range(0, data.data.shape[0], 4):
        if i + 4 > data.data.shape[0]:
            break

        random_indices = outer_random_indices[i : i + 4]
        random_images = data.data[random_indices]
        random_labels = data.targets[random_indices]

        top_row = torch.cat((random_images[0], random_images[1]), dim=0)
        bottom_row = torch.cat((random_images[2], random_images[3]), dim=0)

        img_grid = torch.cat((top_row, bottom_row), dim=1)

        patches = create_patches(img_grid)
        input_labels = tokenize_input_labels(random_labels)
        output_labels = tokenize_output_labels(random_labels)
        yield patches, input_labels, output_labels


def flatten_patches(patches):
    flat_patches = patches.reshape(16, -1)
    return flat_patches


if __name__ == "__main__":
    for grid, labels in generate_grids(train_data):
        patches = create_patches(grid)
        flat_patches = flatten_patches(patches)
        visualize(grid, labels, patches, flat_patches)

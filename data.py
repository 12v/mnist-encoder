import torch
from torchvision import datasets, transforms

from visualization import visualize

train_data = datasets.MNIST(
    root="./data", train=True, download=True, transform=transforms.ToTensor()
)

train_data.data = train_data.data / 255

test_data = datasets.MNIST(
    root="./data", train=False, download=True, transform=transforms.ToTensor()
)

test_data.data = test_data.data / 255


def get_grids(data):
    outer_random_indices = torch.randperm(data.data.shape[0])
    grids = []
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

        grids.append(img_grid)
        labels.append(random_labels)

    return grids, labels


def create_patches(grid):
    patch_count = 4
    patch_size = grid.shape[0] // patch_count

    grid = grid.unfold(0, patch_size, patch_size)
    grid = grid.unfold(1, patch_size, patch_size)

    patches = grid.reshape(-1, patch_size, patch_size)

    return patches


def flatten_patches(patches):
    flat_patches = patches.reshape(16, -1)
    return flat_patches


if __name__ == "__main__":
    grids, labels = get_grids(train_data)
    for grid, label in zip(grids, labels):
        patches = create_patches(grid)
        flat_patches = flatten_patches(patches)
        visualize(grid, label, patches, flat_patches)

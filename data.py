import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms

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


def visualize(grid, labels, patches, flat_patches):
    def visualize_grid(grid, labels):
        plt.imshow(grid, cmap="gray")

        i = 0
        grid_size = grid.shape[0]
        for c in [0, 1]:
            for r in [0, 1]:
                x_coord = c * grid_size / 2
                y_coord = r * grid_size / 2
                plt.text(
                    x_coord,
                    y_coord,
                    str(labels[i].item()),
                    ha="left",
                    va="top",
                    fontsize=24,
                    color="white",
                )
                i += 1
        plt.axis("off")

    def visualize_patches(patches):
        patches = patches.permute(1, 0, 2)
        patches = patches.reshape(14, 224)
        plt.imshow(patches, cmap="gray")
        plt.axis("off")

    def visualize_flat_patches(flat_patches):
        plt.imshow(flat_patches, cmap="gray")
        plt.axis("off")

    plt.figure()
    plt.subplot(3, 1, 1)
    visualize_grid(grid, labels)
    plt.subplot(3, 1, 2)
    visualize_patches(patches)
    plt.subplot(3, 1, 3)
    visualize_flat_patches(flat_patches)
    plt.show()


def visualize_grid(grid):
    plt.imshow(grid, cmap="gray")
    plt.axis("off")
    plt.show()


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


if __name__ == "__main__":
    for grid, labels in generate_grids(train_data):
        patches = create_patches(grid)
        flat_patches = flatten_patches(patches)
        visualize(grid, labels, patches, flat_patches)

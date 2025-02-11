import torch
from torchvision import datasets, transforms


def normalize_and_standardize(data):
    data = (data - 127.5) / 127.5
    data = (data - data.mean()) / data.std()
    return data


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

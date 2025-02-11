from mnist import get_images_and_labels, train_data
from visualization import visualize


def create_patches(image):
    patch_count = 4
    patch_size = image.shape[0] // patch_count

    image = image.unfold(0, patch_size, patch_size)
    image = image.unfold(1, patch_size, patch_size)

    patches = image.reshape(-1, patch_size, patch_size)

    return patches


def flatten_patches(patches):
    flat_patches = patches.reshape(16, -1)
    return flat_patches


if __name__ == "__main__":
    images, labels = get_images_and_labels(train_data)
    for image, label in zip(images, labels):
        patches = create_patches(image)
        flat_patches = flatten_patches(patches)
        visualize(image, label, patches, flat_patches)

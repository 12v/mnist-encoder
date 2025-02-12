def create_patches(image, patch_dim):
    patch_size = image.shape[0] // patch_dim

    image = image.unfold(0, patch_size, patch_size)
    image = image.unfold(1, patch_size, patch_size)

    patches = image.reshape(-1, patch_size, patch_size)

    return patches


def flatten_patches(patches, patch_dim):
    patch_count = patch_dim * patch_dim
    flat_patches = patches.reshape(patch_count, -1)
    return flat_patches


def normalize_and_standardize(data):
    data = data.float()
    data = data - data.mean() / data.std()
    data = 2 * (data - data.min()) / (data.max() - data.min()) - 1
    return data

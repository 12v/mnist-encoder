def create_patches(image, patch_dim):
    num_dims = image.dim()

    if num_dims == 3:
        patch_size = image.shape[1] // patch_dim

        image = image.unfold(1, patch_size, patch_size)
        image = image.unfold(2, patch_size, patch_size)

        patches = image.reshape(3, -1, patch_size, patch_size)
        patches = patches.permute(1, 0, 2, 3)

    elif num_dims == 2:
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

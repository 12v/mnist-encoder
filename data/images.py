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

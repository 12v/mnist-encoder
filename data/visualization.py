import matplotlib.pyplot as plt


def visualize(image, labels, patches, flat_patches):
    def visualize_image(image, labels):
        plt.imshow(image, cmap="gray", vmin=0, vmax=1)

        i = 0
        image_size = image.shape[0]
        print(labels)
        for r in [0, 1]:
            for c in [0, 1]:
                x_coord = c * image_size / 2
                y_coord = r * image_size / 2
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
        plt.axis("off")
        for i in range(16):
            plt.subplot(1, 16, i + 1)
            patch = patches[i]
            plt.imshow(patch, cmap="gray", vmin=0, vmax=1)
            plt.axis("off")

    def visualize_flat_patches(flat_patches):
        plt.axis("off")
        for i in range(16):
            plt.subplot(16, 1, i + 1)
            plt.imshow(flat_patches[i].unsqueeze(0), cmap="gray", vmin=0, vmax=1)
            plt.axis("off")

    plt.figure()
    plt.subplot(3, 1, 1)
    visualize_image(image, labels)
    plt.subplot(3, 1, 2)
    visualize_patches(patches)
    plt.subplot(3, 1, 3)
    visualize_flat_patches(flat_patches)
    plt.show()


def visualize_image(image):
    plt.imshow(image, cmap="gray")
    plt.axis("off")
    plt.show()


def visualize_attention(image, encoder_self_attention, self_attention, cross_attention):
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))

    im = axes[0, 0].imshow(image)
    axes[0, 0].set_title("Input image", fontsize=12)
    axes[0, 0].axis("off")
    fig.colorbar(im, ax=axes[0, 0], fraction=0.046, pad=0.04)

    im = axes[1, 0].imshow(self_attention.squeeze(0))
    axes[1, 0].set_title("Decoder self-attention", fontsize=12)
    axes[1, 0].axis("off")
    fig.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)

    im = axes[0, 1].imshow(encoder_self_attention.squeeze(0))
    axes[0, 1].set_title("Encoder self-attention", fontsize=12)
    axes[0, 1].axis("off")
    fig.colorbar(im, ax=axes[0, 1], fraction=0.046, pad=0.04)

    im = axes[1, 1].imshow(cross_attention.squeeze(0))
    axes[1, 1].set_title("Decoder cross-attention", fontsize=12)
    axes[1, 1].axis("off")
    fig.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()

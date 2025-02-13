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


def visualize_attention(
    image, encoder_self_attention, self_attention, cross_attention, prediction
):
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8, 8))
    outer = gridspec.GridSpec(2, 2, wspace=0.2, hspace=0.2)

    ax = plt.Subplot(fig, outer[0])
    ax.set_title("Input image" + "\nInference: " + " ".join(prediction), fontsize=12)
    ax.imshow(image)
    ax.axis("off")
    fig.add_subplot(ax)

    ax = plt.Subplot(fig, outer[1])
    ax.set_title("Encoder self-attention", fontsize=12)
    ax.axis("off")
    fig.add_subplot(ax)
    inner = gridspec.GridSpecFromSubplotSpec(
        4, 4, subplot_spec=outer[1], wspace=0.1, hspace=0.1
    )
    for i in range(16):
        ax = plt.Subplot(fig, inner[i])
        ax.imshow(encoder_self_attention.squeeze(0).sum(dim=0)[i].reshape(4, 4))
        ax.axis("off")
        fig.add_subplot(ax)

    ax = plt.Subplot(fig, outer[2])
    ax.set_title("Decoder self-attention", fontsize=12)
    ax.imshow(self_attention.squeeze(0).sum(dim=0))
    ax.axis("off")
    fig.add_subplot(ax)

    ax = plt.Subplot(fig, outer[3])
    ax.set_title("Decoder cross-attention", fontsize=12)
    ax.axis("off")
    fig.add_subplot(ax)
    inner = gridspec.GridSpecFromSubplotSpec(
        3, 2, subplot_spec=outer[3], wspace=0.1, hspace=0.1
    )
    for i in range(5):
        ax = plt.Subplot(fig, inner[i])
        ax.imshow(cross_attention.squeeze(0).sum(dim=0)[i].reshape(4, 4))
        ax.axis("off")
        fig.add_subplot(ax)
    plt.show()

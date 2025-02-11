import matplotlib.pyplot as plt


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

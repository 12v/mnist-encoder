import matplotlib.pyplot as plt


def visualize(grid, labels, patches, flat_patches):
    def visualize_grid(grid, labels):
        plt.imshow(grid, cmap="gray", vmin=0, vmax=1)

        i = 0
        grid_size = grid.shape[0]
        print(labels)
        for r in [0, 1]:
            for c in [0, 1]:
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

if __name__ == "__main__":
    # TODO
    import numpy as np
    from base import new_2d_axes, new_3d_axes, plt

    ax1 = new_2d_axes()

    center = np.array([0, 0, 0])
    radius = 1

    draw_sphere(ax1, center, radius)

    plt.show()

    ax2 = new_3d_axes()

    draw_sphere(ax2, center, radius)

    plt.show()

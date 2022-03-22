from utils import drawer


@drawer
def set_limits(ax, *limits):
    pass


@set_limits.dim(2)
def set_2d_limits(ax, *limits):
    ax.set_xlim(*limits[0])
    ax.set_ylim(*limits[1])


@set_limits.dim(3)
def set_3d_limits(ax, *limits):
    ax.set_xlim(*limits[0])
    ax.set_ylim(*limits[1])
    ax.set_zlim(*limits[2])


@drawer
def set_view(ax, *args):
    pass


@set_view.dim(2)
def set_2d_viez(ax, *args):
    raise NotImplementedError("`set_view` is not implemented for 2d axes")


@set_view.dim(3)
def set_3d_view(ax, azim=-60, elev=30):
    return ax.view_init(azim=azim, elev=elev)


if __name__ == "__main__":
    import numpy as np
    from base import new_2d_axes, new_3d_axes, plt
    from polygon import draw_polygon

    ax1 = new_2d_axes()

    points = np.array(
        [
            [0, 0, 0],
            [0, 1, 1],
            [1, 1, 0],
        ]
    )

    draw_polygon(ax1, points)
    set_limits(ax1, [-2.5, 2.5], [-1, 2])

    plt.show()

    ax2 = new_3d_axes()

    draw_polygon(ax2, points)
    set_limits(ax2, [-2.5, 2.5], [-1, 2], [-2, 2])
    set_view(ax2, 0, 90)

    plt.show()

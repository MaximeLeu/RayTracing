from utils import drawer


@drawer
def draw_line(ax, points, *args, **kwargs):
    pass


@draw_line.dim(2)
def draw_2d_line(ax, points, *args, **kwargs):
    """
    Adds a line to 2D axes.

    :param ax: the axes
    :type ax: matplotlib.axes.Axes
    :param points: the points
    :type points: numpy.ndarray *shape=(N, M>=2)*
    :param args: positional arguments passed to :func:`matplotlib.axes.Axes.plot`
    :type args: any
    :param kwargs: keyword arguments passed to :func:`matplotlib.axes.Axes.plot`
    :type kwargs: any
    :return: the lines
    :rtype: List[matplotlib.lines.Line2D]
    """
    x = points[:, 0]
    y = points[:, 1]
    return ax.plot(x, y, *args, **kwargs)


@draw_line.dim(3)
def draw_3d_line(ax, points, *args, **kwargs):
    """
    Adds a set of points to 3D axes.

    :param ax: the axes
    :type ax: mpl_toolkits.mplot3d.Axes3D
    :param points: the points
    :type points: numpy.ndarray *shape=(N, M>=3)*
    :param args: positional arguments passed to :func:`mpl_toolkits.mplot3d.Axes3D.plot`
    :type args: any
    :param kwargs: keyword arguments passed to :func:`mpl_toolkits.mplot3d.Axes3D.plot`
    :type kwargs: any
    :return: the lines
    :rtype: List[mpl_toolkits.mplot3d.art3d.Line3D]
    """
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    return ax.plot(x, y, z, *args, **kwargs)


if __name__ == "__main__":
    from base import new_2d_axes, new_3d_axes, plt
    import numpy as np

    ax1 = new_2d_axes()

    points = np.array(
        [
            [0, 0, 0],
            [0, 1, 1],
            [1, 1, 0],
        ]
    )

    draw_line(ax1, points)

    plt.show()

    ax2 = new_3d_axes()

    draw_line(ax2, points)

    plt.show()

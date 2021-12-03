from utils import drawer

import matplotlib.patches as patches


@drawer
def draw_sphere(ax, center, radius, *args, **kwargs):
    pass


@draw_sphere.dim(2)
def draw_2d_sphere(ax, center, radius, *args, **kwargs):
    """
    TODO
    Adds a 2D vector to axes. If higher dimension vector is given, will only keep the 2 first dimensions.

    :param ax: axes
    :type ax: matplotlib.axes.Axes
    :param point: the origin of the vector
    :type point: numpy.ndarray *shape(N, M>=2)*
    :param vector: the direction of the vector
    :type vector: numpy.ndarray *shape(N, M>=2)*
    :param args: positional arguments passed to :func:`matplotlib.pyplot.quiver`
    :type args: any
    :param kwargs: keyword arguments passed to :func:`matplotlib.pyplot.quiver`
    :type kwargs: any
    :return: the line collection
    :rtype: matplotlib.collections.LineCollection
    """
    xy = center[:2].flat
    return ax.add_patch(patches.Circle(xy, radius, *args, **kwargs))


@draw_sphere.dim(3)
def draw_3d_sphere(ax, center, radius, *args, **kwargs):
    """
    TODO
    Adds a 3D vector to axes. If higher dimension vector is given, will only keep the 3 first dimensions.

    :param ax: axes
    :type ax: mpl_toolkits.mplot3d.Axes3D
    :param point: the origin of the vector
    :type point: numpy.ndarray *shape(N, M>=3)*
    :param vector: the direction of the vector
    :type vector: numpy.ndarray *shape(N, M>=3)*
    :param args: positional arguments passed to :func:`mpl_toolkits.mplot3d.Axes3D.quiver`
    :type args: any
    :param kwargs: keyword arguments passed to :func:`mpl_toolkits.mplot3d.Axes3D.quiver`
    :type kwargs: any
    :return: the line collection
    :rtype: mpl_toolkits.mplot3d.art3d.Line3DCollection
    """
    u, v = np.mgrid[0 : 2 * np.pi : 50j, 0 : np.pi : 25j]
    x = radius * np.cos(u) * np.sin(v) + center[0]
    y = radius * np.sin(u) * np.sin(v) + center[1]
    z = radius * np.cos(v) + center[2]
    return ax.plot_surface(
        x, y, z, color="r", rstride=1, cstride=1, linewidth=0, antialiased=False
    )


if __name__ == "__main__":
    from base import new_2d_axes, new_3d_axes, plt
    import numpy as np

    ax1 = new_2d_axes()

    center = np.array([0, 0, 0])
    radius = 1

    draw_sphere(ax1, center, radius)

    plt.show()

    ax2 = new_3d_axes()

    draw_sphere(ax2, center, radius)

    plt.show()

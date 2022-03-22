import matplotlib.patches as patches
import mpl_toolkits.mplot3d as mplot3d

from .utils import drawer


@drawer
def draw_polygon(ax, points, *args, **kwargs):
    pass


@draw_polygon.dim(2)
def draw_2d_polygon(ax, points, *args, **kwargs):
    """
    Adds a 2D polygon to axes. If higher dimension polygon is given, will only keep the 2 first dimensions.

    :param ax: axes
    :type ax: matplotlib.axes.Axes
    :param points: the points of the polygon
    :type points: numpy.ndarray *shape(N, M>=2)*
    :param args: positional arguments passed to :func:`matplotlib.patches.PathPatch`
    :type args: any
    :param kwargs: keyword arguments passed to :func:`matplotlib.patches.PathPatch`
    :type kwargs: any
    :return: the polygon patch
    :rtype: matplotlib.patches.PathPatch
    """
    path = patches.Path(points[:, :2])
    patch = patches.PathPatch(path, *args, **kwargs)
    ax.add_patch(patch)
    return patch


@draw_polygon.dim(3)
def draw_3d_polygon(ax, points, *args, **kwargs):
    """
    Adds a 3D polygon to axes. If higher dimension polygon is given, will only keep the 3 first dimensions.

    :param ax: axes
    :type ax: mpl_toolkits.mplot3d.Axes3D
    :param points: the points of the polygon
    :type points: numpy.ndarray *shape(N, M>=3)*
    :param args: positional arguments passed to :func:`mpl_toolkits.mplot3d.art3d.Poly3DCollection`
    :type args: any
    :param kwargs: keyword arguments passed to :func:`mpl_toolkits.mplot3d.art3d.Poly3DCollection`
    :type kwargs: any
    :return: the line collection
    :rtype: mpl_toolkits.mplot3d.art3d.Poly3DCollection
    """
    polygon = mplot3d.art3d.Poly3DCollection([points], *args, **kwargs)
    ax.add_collection3d(polygon)
    return polygon


if __name__ == "__main__":
    import numpy as np
    from base import new_2d_axes, new_3d_axes, plt

    ax1 = new_2d_axes()

    points = np.array(
        [
            [0, 0, 0],
            [0, 1, 1],
            [1, 1, 0],
        ]
    )

    draw_polygon(ax1, points)

    plt.show()

    ax2 = new_3d_axes()

    draw_polygon(ax2, points)

    plt.show()

# Plotting libraries
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as mplot3d
import matplotlib.patches as patches


def get_2d_plot_ax(ax=None):
    """
    Returns axes on which 2D geometries can be plotted.

    :param ax: If a matplotlib axes given, will return it
    :type ax: matplotlib.axes.Axes
    :returns: axes
    :rtype: matplotlib.axes.Axes
    """
    if ax is None:
        ax = plt.subplot(111)
    return ax


def get_3d_plot_ax(ax=None):
    """
    Returns axes on which 3D geometries can be plotted.

    :param ax: If a matplotlib axes given, will return it
    :type ax: mpl_toolkits.mplot3d.Axes3D
    :returns: axes
    :rtype: mpl_toolkits.mplot3d.Axes3D
    """
    if ax is None:
        ax = mplot3d.Axes3D(fig=plt.figure())
    return ax


def add_polygon_to_2d_ax(ax, points, *args, **kwargs):
    """
    Adds a 2D polygon to axes. If higher dimension polygon is given, will only keep the 2 first dimensions.

    :param ax: axes
    :type ax: matplotlib.axes.Axes
    :param points: the points of the polygon
    :type points: numpy.ndarray *shape(N, M>=2)*
    :param args: positional arguments passed to :func:`matplotlib.patches.Polygon`
    :type args: any
    :param kwargs: keyword arguments passed to :func:`matplotlib.patches.Polygon`
    :type kwargs: any
    """
    polygon = patches.Polygon(points[:, :2], *args, **kwargs)
    ax.add_patch(polygon)


def add_polygon_to_3d_ax(ax, points, *args, **kwargs):
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
    """
    polygon = mplot3d.art3d.Poly3DCollection([points], *args, **kwargs)
    ax.add_collection3d(polygon)


def add_vector_to_2d_ax(ax, point, vector, *args, **kwargs):
    """
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
    """
    x, y = point[:2]
    u, v = vector[:2]
    ax.quiver(x, y, u, v, *args, **kwargs)


def add_vector_to_3d_ax(ax, point, vector, *args, **kwargs):
    """
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
    """
    x, y, z = point
    u, v, w = vector
    ax.quiver(x, y, z, u, v, w, *args, **kwargs)

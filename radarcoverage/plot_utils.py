# Plotting libraries
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as mplot3d
import matplotlib.patches as patches

# Numerical libraries
import numpy as np


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


def add_points_to_2d_ax(ax, points, *args, **kwargs):
    points = points.reshape(-1, 3)
    ax.scatter(points[:, 0], points[:, 1], *args, **kwargs)


def add_points_to_3d_ax(ax, points, *args, **kwargs):
    points = points.reshape(-1, 3)
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], *args, **kwargs)


def add_line_to_2d_ax(ax, points, *args, **kwargs):
    x = points[:, 0]
    y = points[:, 1]
    ax.plot(x, y, *args, **kwargs)


def add_line_to_3d_ax(ax, points, *args, **kwargs):
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    ax.plot(x, y, z, *args, **kwargs)


def add_text_at_point_2d_ax(ax, point, text, *args, **kwargs):
    point = point.reshape(2)
    ax.annotate(text, point)


def add_text_at_point_3d_ax(ax, point, text, *args, **kwargs):
    point = point.reshape(3)
    ax.text(point[0], point[1], point[2], text, *args, **kwargs)


def add_2d_text_at_point_3d_ax(ax, point, text, *args, **kwargs):
    point = point.reshape(-1)
    ax.text2D(point[0], point[1], text, *args, transform=ax.transAxes, **kwargs)


def add_polygon_to_2d_ax(ax, points, *args, **kwargs):
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
    """
    path = patches.Path(points[:, :2])
    patch = patches.PathPatch(path, *args, **kwargs)
    ax.add_patch(patch)


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
    x, y = point[:2].T
    u, v = vector[:2].T
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
    x, y, z = point.T
    u, v, w = vector.T
    ax.quiver(x, y, z, u, v, w, *args, **kwargs)

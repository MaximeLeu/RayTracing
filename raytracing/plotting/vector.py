from .utils import drawer


@drawer
def draw_vector(ax, point, vector, *args, **kwargs):
    pass


@draw_vector.dim(2)
def draw_2d_vector(ax, point, vector, *args, **kwargs):
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
    :return: the line collection
    :rtype: matplotlib.collections.LineCollection
    """
    x, y = point[:2].flat
    u, v = vector[:2].flat
    return ax.quiver(x, y, u, v, *args, **kwargs)


@draw_vector.dim(3)
def draw_3d_vector(ax, point, vector, *args, **kwargs):
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
    :return: the line collection
    :rtype: mpl_toolkits.mplot3d.art3d.Line3DCollection
    """
    x, y, z = point.flat
    u, v, w = vector.flat
    return ax.quiver(x, y, z, u, v, w, *args, **kwargs)

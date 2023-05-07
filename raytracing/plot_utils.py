#pylint: disable=invalid-name,line-too-long

# Utils
import sys
# Plotting libraries
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import patches
# Numerical libraries
import numpy as np
import shapely

anim = dict(
    pause=False,
    speed=1
)

def ensure_axis_orthonormal(ax):
    ax.set_box_aspect([1,1,1])
    ax.set_proj_type('ortho')
    ax.axis('equal')
    return ax
    

def plot_shapely(shapely_object,ax=None,color=None):
    #https://coderslegacy.com/python/plotting-shapely-polygons-with-interiors-holes/
    if ax is None:
        fig, ax = plt.subplots()
    if isinstance(shapely_object, shapely.geometry.Point):
        ax.plot(shapely_object.x, shapely_object.y, marker='o', markersize=5)
        
    elif isinstance(shapely_object, shapely.geometry.Polygon):
        if color is None:
            x, y = shapely_object.exterior.xy
            ax.plot(x, y)
            for inner in shapely_object.interiors:
                xi, yi = zip(*inner.coords[:])
                ax.plot(xi, yi)
        else:
            x, y = shapely_object.exterior.xy
            ax.plot(x, y,color='black')
            for inner in shapely_object.interiors:
                xi, yi = zip(*inner.coords[:])
                ax.plot(xi, yi,color=color)
            
        
            
    elif isinstance(shapely_object, shapely.geometry.MultiPolygon):
        for obj in shapely_object.geoms:
            x, y = obj.exterior.xy
            ax.plot(x, y)
            for inner in obj.interiors:
                xi, yi = zip(*inner.coords[:])
                ax.plot(xi, yi)
    else:
        raise ValueError("Unsupported Shapely object type.")

    return ax

def plot_vec(ax,vec,color,origin):
    ax.plot([origin[0],origin[0]+vec[0]],[origin[1],origin[1]+vec[1]],[origin[2],origin[2]+vec[2]],color=color)
    return

def plot_world_frame(ax,colors):
    world_basis=[np.array([1,0,0]),np.array([0,1,0]),np.array([0,0,1])]
    origin=np.array([0,0,0])
    for i in range(3):
        plot_vec(ax,world_basis[i],colors[i],origin=origin)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    return ax

def plot_path(ax,path):
    add_points_to_3d_ax(ax=ax, points=np.array([path[0]]), label="TX")
    add_points_to_3d_ax(ax=ax, points=np.array([path[-1]]), label="RX")
    for i in range(len(path)-1):
        ax.plot([path[i][0],path[i+1][0]],[path[i][1],path[i+1][1]],[path[i][2],path[i+1][2]])
    return ax


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


def set_axes_label(ax, labels):
    """
    Sets the axes labels.

    :param ax: the axes
    :type ax: matplotlib.axes.Axes or mpl_toolkits.mplot3d.Axes3D
    :param labels: the labels to be set
    :type labels: list *len=3*
    """
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    if isinstance(ax, mplot3d.Axes3D):
        ax.set_zlabel(labels[2])


def set_cartesian_axes_label(ax):
    """
    Sets the axes labels to x, y and optionally z if 3D axes are passed.

    :param ax: the axes
    :type ax: matplotlib.axes.Axes or mpl_toolkits.mplot3d.Axes3D
    """
    set_axes_label(ax, ['x', 'y', 'z'])


def set_polar_axes_label(ax):
    """
    Sets the axes labels to r and theta.

    :param ax: the axes
    :type ax: matplotlib.axes.Axes
    """
    set_axes_label(ax, ['r', r'\theta'])


def set_spherical_axes_label(ax):
    """
    Sets the axes labels to r, phi and theta.

    :param ax: the axes
    :type ax: mpl_toolkits.mplot3d.Axes3D
    """
    set_axes_label(ax, ['r', r'\phi', r'\theta'])


def add_points_to_2d_ax(ax, points, *args, **kwargs):
    """
    Adds a set of points to 2D axes.

    :param ax: the axes
    :type ax: matplotlib.axes.Axes
    :param points: the points
    :type points: numpy.ndarray *shape=(N, M>=2)*
    :param args: positional arguments passed to :func:`matplotlib.axes.Axes.scatter`
    :type args: any
    :param kwargs: keyword arguments passed to :func:`matplotlib.axes.Axes.scatter`
    :type kwargs: any
    :return: the path collection
    :rtype: matplotlib.collections.PathCollection
    """
    return ax.scatter(points[:, 0], points[:, 1], *args, **kwargs)


def add_points_to_3d_ax(ax, points, *args, **kwargs):
    """
    Adds a set of points to 3D axes.

    :param ax: the axes
    :type ax: mpl_toolkits.mplot3d.Axes3D
    :param points: the points
    :type points: numpy.ndarray *shape=(N, M>=3)*
    :param args: positional arguments passed to :func:`mpl_toolkits.mplot3d.Axes3D.scatter`
    :type args: any
    :param kwargs: keyword arguments passed to :func:`mpl_toolkits.mplot3d.Axes3D.scatter`
    :type kwargs: any
    :return: the path collection
    :rtype: mpl_toolkits.mplot3d.art3d.Path3DCollection
    """
    #to handle the case when a single point is given.
    if len(points.shape) == 1:
        points = points.reshape(1, -1)
    return ax.scatter(points[:, 0], points[:, 1], points[:, 2], *args, **kwargs)


def add_line_to_2d_ax(ax, points, *args, **kwargs):
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


def add_line_to_3d_ax(ax, points, *args, **kwargs):
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


def add_text_at_point_2d_ax(ax, point, text, *args, **kwargs):
    """
    Adds a text to 2D axes.

    :param ax: the axes
    :type ax: matplotlib.axes.Axes
    :param point: the position of the text
    :type point: numpy.ndarray *size=2*
    :param text: the text to be displayed
    :type text: str
    :param args: positional arguments passed to :func:`matplotlib.axes.Axes.text`
    :type args: any
    :param kwargs: keyword arguments passed to :func:`matplotlib.axes.Axes.text`
    :type kwargs: any
    :return: the annotation
    :rtype: matplotlib.text.Annotation
    """
    return ax.annotate(text, point, *args, **kwargs)


def add_text_at_point_3d_ax(ax, point, text, *args, **kwargs):
    """
    Adds a text to 3D axes.

    :param ax: the axes
    :type ax: mpl_toolkits.mplot3d.Axes3D
    :param point: the position of the text
    :type point: numpy.ndarray *size=3*
    :param text: the text to be displayed
    :type text: str
    :param args: positional arguments passed to :func:`mpl_toolkits.mplot3d.Axes3D.text`
    :type args: any
    :param kwargs: keyword arguments passed to :func:`mpl_toolkits.mplot3d.Axes3D.text`
    :type kwargs: any
    :return: # TODO
    :rtype: # TODO
    """
    point = point.flat
    return ax.text(point[0], point[1], point[2], text, *args, **kwargs)


def add_2d_text_at_point_3d_ax(ax, point, text, *args, **kwargs):
    """
    Adds a 2D text to 3D axes.

    :param ax: the axes
    :type ax: mpl_toolkits.mplot3d.Axes3D
    :param point: the position of the text
    :type point: numpy.ndarray *size=2*
    :param text: the text to be displayed
    :type text: str
    :param args: positional arguments passed to :func:`mpl_toolkits.mplot3d.Axes3D.text2D`
    :type args: any
    :param kwargs: keyword arguments passed to :func:`mpl_toolkits.mplot3d.Axes3D.text2D`
    :type kwargs: any
    :return: # TODO
    :rtype: # TODO

    :Example:

    >>> ax = get_3d_plot_ax()
    >>> pos = np.array([0.05, 0.95])  # North-West position
    >>> add_2d_text_at_point_3d_ax(ax, pos, 'Hello', *args, **kwargs)
    """
    point = point.reshape(-1)
    return ax.text2D(point[0], point[1], text, *args, transform=ax.transAxes, **kwargs)


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
    :return: the polygon patch
    :rtype: matplotlib.patches.PathPatch
    """
    path = patches.Path(points[:, :2])
    patch = patches.PathPatch(path, *args, **kwargs)
    ax.add_patch(patch)
    return patch


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
    :return: the line collection
    :rtype: mpl_toolkits.mplot3d.art3d.Poly3DCollection
    """
    polygon = mplot3d.art3d.Poly3DCollection([points], *args, **kwargs)
    ax.add_collection3d(polygon)
    return polygon


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
    :return: the line collection
    :rtype: matplotlib.collections.LineCollection
    """
    x, y = point[:2].flat
    u, v = vector[:2].flat
    return ax.quiver(x, y, u, v, *args, **kwargs)


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
    :return: the line collection
    :rtype: mpl_toolkits.mplot3d.art3d.Line3DCollection
    """
    x, y, z = point.flat
    u, v, w = vector.flat
    return ax.quiver(x, y, z, u, v, w, *args, **kwargs)


def animate_3d_ax(ax, dt=0.01, func=None, *args, **kwargs):
    """
    Starts a 3D animation where the plot rotates endlessly or until `func` returns 0.

    :param ax: axes
    :type ax: mpl_toolkits.mplot3d.Axes3D
    :param dt: the sleep time in seconds between each frame
    :type dt: float
    :param func: a function that will be called at every loop, must take 3D axes as first argument
    :param args: positional arguments passed to `func`
    :type args: any
    :param kwargs: keyword arguments passed to `func`
    :type kwargs: any
    """
    pos = np.array([0.05, 0.90])

    global anim

    def txt(anim):
        speed = anim['speed']

        return f'Press \'q\' to quit, \'space\' to play/pause,\n'\
               f'\'>\' (\'<\') to accel. (slow) the animation, \'0\' to reset.\n'\
               f'Current speed: {speed} [it./frame]'

    text = add_2d_text_at_point_3d_ax(ax, pos, txt(anim))

    fig = plt.gcf()

    def press(event):
        global anim
        sys.stdout.flush()
        if event.key.lower() == 'q':
            plt.close(fig)
        elif event.key.lower() == ' ':
            anim['pause'] ^= True
        elif event.key.lower() == '>':
            anim['speed'] += 1
        elif event.key.lower() == '<':
            anim['speed'] -= 1
            anim['speed'] = max(0, anim['speed'])
        elif event.key.lower() == '0':
            anim['speed'] = 1

    fig.canvas.mpl_connect('key_press_event', press)

    angle = 0
    ax.view_init(30, 0)

    while plt.fignum_exists(fig.number):

        text.set_text(txt(anim))

        if anim['pause']:
            angle = ax.azim
            plt.pause(0.1)
            continue

        ax.view_init(ax.elev, angle)

        if func is not None:
            for _ in range(anim['speed']):
                ret = func(ax, *args, **kwargs)
                if ret == 0:
                    break

        plt.draw()
        plt.pause(dt)
        angle = (ax.azim + 1) % 360

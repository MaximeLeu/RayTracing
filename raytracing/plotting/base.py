import matplotlib.pyplot as plt
import numpy as np

from .misc import set_limits


def __new_axes(*args, **kwargs):
    fig, ax = plt.subplots(*args, **kwargs)
    return ax


def new_2d_axes(*args, **kwargs):
    return __new_axes(*args, **kwargs)


def new_3d_axes(*args, **kwargs):
    kwargs["subplot_kw"] = {**kwargs.get("subplot_kw", {}), "projection": "3d"}
    return __new_axes(*args, **kwargs)


class Plotable(object):
    def __init__(self, ax=None, dim=3):
        assert 2 <= dim <= 3
        self.dim = dim
        self._ax = ax

    @property
    def ax(self):
        if self._ax is None:
            if self.dim == 2:
                self._ax = new_2d_axes()
            else:
                self._ax = new_3d_axes()
        return self._ax

    def on(self, ax):
        self._ax = ax
        return self

    def set_limits(self, domain, centroid=np.array([0, 0, 0]), kind="equal"):
        if kind == "equal":
            bound = np.max(domain[1] - domain[0])
            limits = np.vstack((centroid - bound / 2, centroid + bound / 2))

        elif kind == "tight":
            limits = domain

        set_limits(self.ax, *limits.T)

    def show(self):
        plt.show()

    def plot(self, *args, **kwargs):
        raise NotImplementedError

    def plot2d(self, *args, ax=None, **kwargs):
        """
        Plots the geometry on 2D axes using z=0 projection.

        :param args: geometry-specific positional arguments
        :type args: any
        :param ax: optionally, the axes on which to plot
        :type ax: matplotlib.axes.Axes
        :param kwargs: geometry-specific keyword arguments
        :type kwargs: any
        :return: the axes
        :rtype: matplotlib.axes.Axes
        """
        raise NotImplementedError

    def plot3d(self, *args, ax=None, **kwargs):
        """
        Plots the geometry on 3D axes.

        :param args: geometry-specific positional arguments
        :type args: any
        :param ax: optionally, the axes on which to plot
        :type ax: mpl_toolkits.mplot3d.Axes3D
        :param kwargs: geometry-specific keyword arguments
        :type kwargs: any
        :return: the axes
        :rtype: mpl_toolkits.mplot3d.Axes3D
        """
        raise NotImplementedError

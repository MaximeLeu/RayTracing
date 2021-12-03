import matplotlib.pyplot as plt


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
        if ax:
            self.ax = ax
        else:
            if dim == 2:
                self.ax = new_2d_axes()
            elif dim == 3:
                self.ax = new_3d_axes()
            pass
        raise NotImplementedError

    def plot(self, *args, ax=None, **kwargs):
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

from functools import wraps
from collections import defaultdict

from matplotlib.axes import Axes as Axes2D
from mpl_toolkits.mplot3d import Axes3D


def axes_dim(ax):
    if isinstance(ax, Axes3D):
        return 3
    elif isinstance(ax, Axes2D):
        return 2
    else:
        raise ValueError("Unsupported type of axes `%s`" % ax)


class Drawer(object):
    def __init__(self, default):
        self.__default = default
        self.__implementations = defaultdict(lambda: default)

    def __repr__(self):
        return "<Drawer(%s)>" % self.__name__

    def set_impl(self, dim, impl):
        self.__implementations[dim] = impl

    def get_impl(self, dim):
        return self.__implementations[dim]

    def dim(self, dim):
        def _impl_(func):
            @wraps(func)
            def __wrapper__(*args, **kwargs):
                return func(*args, **kwargs)

            self.set_impl(dim, __wrapper__)
            return __wrapper__

        return _impl_

    def __call__(self, ax, *args, **kwargs):
        dim = axes_dim(ax)
        impl = self.get_impl(dim)
        return impl(ax, *args, **kwargs)


def drawer(func):
    return wraps(func)(Drawer(func))

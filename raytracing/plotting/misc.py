from .utils import drawer


@drawer
def set_limits(ax, *limits):
    pass


@set_limits.dim(2)
def set_2d_limits(ax, *limits):
    ax.set_xlim(*limits[0])
    ax.set_ylim(*limits[1])


@set_limits.dim(3)
def set_3d_limits(ax, *limits):
    ax.set_xlim(*limits[0])
    ax.set_ylim(*limits[1])
    ax.set_zlim(*limits[2])


@drawer
def set_view(ax, *args):
    pass


@set_view.dim(2)
def set_2d_view(ax, *args):
    raise NotImplementedError("`set_view` is not implemented for 2d axes")


@set_view.dim(3)
def set_3d_view(ax, azim=-60, elev=30):
    return ax.view_init(azim=azim, elev=elev)

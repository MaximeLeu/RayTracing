from .line import draw_line


def draw_points(ax, point, *args, **kwargs):
    return draw_line(ax, point, ".", *args, **kwargs)

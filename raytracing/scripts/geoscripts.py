import click
from click_repl import register_repl
from raytracing import geometry as geom
from raytracing import plot_utils
import matplotlib.pyplot as plt
import numpy as np


@click.group()
def geometry():
    """
    Tools to manipulate geometries.

    If in REPL, use CTRL+D to exit.
    """


@geometry.command()
@click.argument("i", type=click.Path(exists=True))
@click.argument("o", type=click.Path())
@click.option("--geotype", default="place", help="Type of geometry to be loaded")
def create(i, o, geotype):
    """
    Creates a new geometry from .geojson file.

    I is the input .geogson file
    O is the output .ogeom file
    GEOTYPE is the type of geometry to be created
    """
    if geotype == "place":
        place = geom.generate_place_from_rooftops_file(i)
        place.save(o)
        click.secho(f"Successfully created and saved geometry to {o}", fg="green")
    else:
        click.secho(f"Type {geotype} not currently supported.", fg="red")


@geometry.command()
@click.argument("i", type=click.Path(exists=True))
@click.argument("t", type=int, nargs=3)
@click.option(
    "--o",
    type=click.Path(),
    default=None,
    help="Output .ogeom file, same as input file if not specified",
)
def translate(i, t, o):
    """
    Translates a geometry with a given displacement vector.

    I is the input .ogeom file
    T is the translation vector
    """
    if o is None:
        o = i

    g = geom.OrientedGeometry.load(i)
    vector = np.array(t)
    g.translate(vector).save(o)

    click.secho(
        f"Successfully translated geometry in {i} by {vector.tolist()} and saved the result in {o}",
        fg="green",
    )


@geometry.command()
@click.argument("i", type=click.Path(exists=True))
@click.option("--dim", default=3, type=int, help="2 or 3 for 2d or 3d plot.")
def show(i, dim):
    """
    Shows a geometry in a 2D or 3D plot.

    I is the input .ogeom file
    DIM allows to choose between 2D or 3D plot
    """
    g = geom.OrientedGeometry.load(i)
    if dim == 2:
        ax = g.plot2d(ret=True)
        plot_utils.set_cartesian_axes_label(ax)
        g.center_2d_plot(ax)
        plt.show()
    elif dim == 3:
        ax = g.plot3d(ret=True)
        plot_utils.set_cartesian_axes_label(ax)
        g.center_3d_plot(ax)
        plt.show()
    else:
        click.secho(f"Cannot plot in {dim}D.", fg="red")


register_repl(geometry)

import click
from click_repl import register_repl
from radarcoverage import geometry as geom
import matplotlib.pyplot as plt


@click.group()
def geometry():
    """
    Tools to manipulate geometries.

    If in REPL, use CTRL+D to exit.
    """


@geometry.command()
@click.argument('i', type=click.Path(exists=True))
@click.argument('o', type=click.Path())
@click.option('--geotype', default='place', help='Type of geometry to be loaded')
def create(i, o, geotype):
    """
    Creates a new geometry from .geojson file.

    I is the input .geogson file
    O is the output .ogeom file
    GEOTYPE is the type of geometry to be created
    """
    if geotype == 'place':
        place = geom.generate_place_from_rooftops_file(i)
        place.save(o)
        click.secho(f'Successfully created and saved geometry to {o}', fg='green')
    else:
        click.secho(f'Type {geotype} not currently supported.', fg='red')


@geometry.command()
@click.argument('i', type=click.Path(exists=True))
@click.option('--dim', default=3, type=int, help='2 or 3 for 2d or 3d plot.')
def show(i, dim):
    """
    Shows a geometry in a 2D or 3D plot.

    I is the input .ogeom file
    DIM allows to choose between 2D or 3D plot
    """
    g = geom.OrientedGeometry.load(i)
    if dim == 2:
        g.plot2d()
        plt.show()
    elif dim == 3:
        ax = g.plot3d(ret=True)
        g.center_3d_plot(ax)
        plt.show()
    else:
        click.secho(f'Cannot plot in {dim}D.', fg='red')


register_repl(geometry)

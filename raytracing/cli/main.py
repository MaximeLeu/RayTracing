import click
from click_repl import register_repl
from geometry import geometry


@click.group()
def cli():
    """Provide RayTracing command-line utilities"""
    pass


cli.add_command(geometry)

register_repl(cli)


if __name__ == "__main__":
    cli()

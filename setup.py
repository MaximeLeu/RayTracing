from setuptools import setup, find_packages

setup(
    name='radarcoverage',
    version='1.0',
    packages=find_packages(),
    include_package_date=True,
    install_requires=[
        'Click'
    ],
    entry_points="""
        [console_scripts]
        geometry=radarcoverage.scripts.geometry:geometry
    """
)
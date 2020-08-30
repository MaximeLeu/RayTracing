from setuptools import setup, find_packages

with open('requirements.txt', 'r') as f:
    install_requires = f.read()

setup(
    name='radarcoverage',
    version='1.0',
    packages=find_packages(),
    include_package_date=True,
    install_requires=install_requires,
    entry_points="""
        [console_scripts]
        geometry=radarcoverage.scripts.geoscripts:geometry
    """
)
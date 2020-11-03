from setuptools import setup, find_packages

setup(
    name='raytracing',
    author='Jérome Eertmans',
    author_email='jerome.eertmans@student.uclouvain.be|jeertmans@icloud.com',
    version='1.0',
    packages=find_packages(),
    include_package_date=True,
    entry_points="""
        [console_scripts]
        geometry=raytracing.scripts.geoscripts:geometry
    """
)
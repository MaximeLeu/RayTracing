# Ray Tracing
This project is the subject of a thesis and an article:
 - The article, detailing the ray tracing method, called min path tracing is available [here](https://doi.org/10.48550/arXiv.2301.06399)
 - The thesis, detailing the electromagnetism computations, will be available on [UCLouvain's dial.mem](https://dial.uclouvain.be/memoire/ucl/fr/search/site) under the name "Modeling complex objects within ray tracing simulations applied to beyond 5G communications", written by Maxime Leurquin

This project's main objective is to provide intuitive tools in order to compute ray tracing in cities. It takes into account multiple orders of reflections and/or refraction for a given signal path. When possible, analytical solutions are provided, but numerical solutions are also used in order to reduce the complexity of the software.

## Project structure

*Only main folders and files are shown.*
```
Radar coverage
│   README.md
└───raytracing - Code to compute the paths the rays take.
└───electromagnetism_fun - Code to compute the electromagnetic field propagated by the rays.
└───examples - visual examples
└───docs - documentation sources and builds
    │   README.MD
└───tests - tests used for C.I.
```

## Setup your project

### On Linux or MacOS machines
1. Clone this repo
2. Install Python 3.10 (or higher)
3. Setup a virtual env. [with pip](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/) (recommended but not mandatory)
4. Install required packages: 
```
pip3 install -r requirements.txt
pip3 install -e .
```

### On Windows machines
1. Clone this repo
2. Install Python 3.6 (or higher) via [anaconda](https://www.anaconda.com/products/individual). While it is possible to do everything without Anaconda, I highly recommend using it because it will make your life much simpler and avoid many installation problems...
3. The rest of the tutorial will be done using the **Anaconda Prompt**
4. Setup a virtual environment with [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html):
```
conda create --name RayTracing
conda activate RayTracing
```
5. Install the packages using:
```
setup
```


*Note: in order to accelerate the computation and make it more robust, this project uses Numba's JIT compiler.
While it provides good performances, you may observe a small compilation time before any program execution.*


## Obtaining data files

This project examples use open source data from OpenStreetMap, using [overpass turbo](https://overpass-turbo.eu/).
Data may vary from location - not all locations are documented the same way - and a tutorial explaining how to setup the data is provided [here](/data/README.md).

Alternatively, you can use software such as [QGIS](https://www.qgis.org), which is Open Source, in order to manipulate geographical data files and create the required geometry.

## Test your project

Just run the follow command:

```
pytest
```

## TODO-list

- [] provide more tests
- [] document all functions
- [] Improve ray tracing performance
- [] Load RayTracingProblem from file
- [] use Numba's ahead of time compiler to remove overhead time for each compilation, [see here](https://numba.pydata.org/numba-doc/dev/user/pycc.html)

## Documentation

The documentation of this project is generated using *Sphinx*. More information about how to proceed in the [**docs/README**](/docs/README.md).


## Known issues
1. If the code does not end and the processors are not under load, it is probably because the code was launched in an IPython environment. IPython has trouble with the multiprocessing implementation. Try running the code from the console directly, or restart your kernel before each launch in an IPython environment.
2. If you renamed one of the project folders, you may encounter an error because old *cached data* (e.g. from Numba)
does not match new paths. To solve this, delete all the `__pycache__` folders.
3. If you encounter this error `no arguments in initialization list` and that you are running on Windows,
please try the solutions mentioned [here](https://github.com/pyproj4/pyproj/issues/134#issuecomment-458813395).
4. If you encounter this error:
```
A GDAL API version must be specified. Provide a path to gdal-config using a GDAL_CONFIG environment variable or use a GDAL_VERSION environment variable.
    ----------------------------------------
ERROR: Command errored out with exit status 1: python setup.py egg_info Check the logs for full command output.)
```
then gdal has trouble getting installed. Try `pip3 install gdal`. If an error occurs saying you need Microsoft Visual C++ 14.0, then install it from [here](https://visualstudio.microsoft.com/visual-cpp-build-tools/).


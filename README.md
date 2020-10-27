# Radar coverage

This project's main objective is to provide intuitive tools in order to compute radar coverage in cities. It takes into account multiple orders of reflections and/or refraction for a given signal path. When possible, analytical solutions are provided, but numerical solutions are also used in order to reduce the complexity of the software.

## Project structure

*Only main folders and files are shown.*
```
Radar coverage
│   README.md
└───radarcoverage - code
└───examples - visual exampes
└───docs - documentation sources and builds
    │   README.MD
└───tests - tests used for C.I.
```

## Setup your project

1. Clone this repo
2. Install Python 3.6 (or higher)
3. [Setup a virtual env](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/) (recommended but not mandatory)
4. Install required packages: `pip3 install install -e .` (**mandatory to enable command-line tools**)


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
- [] add material information (reflection coefficient, etc.)
- [] validate power calculation
- [] validate diffraction E.M. calculation

## Documentation

The documentation of this project is generated using *Sphinx*. More information about how to proceed in the [**docs/README**](/docs/README.md).

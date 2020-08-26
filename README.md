# Radar coverage

This project's main objective is to provide intuitive tools in order to compute radar coverage in cities. In take into account multiple orders of reflections and/or refraction for a given signal path. When possible, analytical solutions are provided, but numerical solutions are also used in order to reduce the complexity of the software.

## Project structure

*Only main folders and files are shown.*
```
Radar coverage
│   README.md
└───radarcoverage
└───examples - visual exampes
└───docs - documentation sources and builds
    │   README.MD
```

## Setup your project

1. Clone this repo
2. Install Python 3.6 (or higher)
3. [Setup a virtual env](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/) (advised but not mandatory)
4. Install required packages: `pip3 install -r requirements.txt`

## Obtaining data files

This project examples use open source data from OpenStreetMap, using [overpass turbo](https://overpass-turbo.eu/).
Data may vary from location - not all locations are documented the same way - and a tutorial explaining how to setup the data is provided [here](todo).

## Documentation

The documentation of this project is generated using *Sphinx*. More information about how to proceed in the [**README**](/docs/README.md).

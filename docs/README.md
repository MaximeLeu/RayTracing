# Documentation

The documentation of this project tries to follow the guidelines imposed by the Python foundation ([pep8](https://www.python.org/dev/peps/pep-0008/)).

## Setup

To be able to compile this documentation, you will need to install these packages:

```
pip3 install sphinx
pip3 install sphinx-rtd-theme
```

## Add files to documentation
If you created a new file which has not been documented yet, you need to include it in the source directory:
```
sphinx-apidoc -o source/radarcoverage ../radarcoverage
```

## Build

After any change in the code, in order to see the changes:
```
make html
```

The index page of the project's documentation will be [here](build/html/index.html).

## Additional content

Here are some interesting links:
- https://sphinx-rtd-tutorial.readthedocs.io/en/latest/docstrings.html
- https://thomas-cokelaer.info/tutorials/sphinx/docstring_python.html
- https://thomas-cokelaer.info/tutorials/sphinx/rest_syntax.html
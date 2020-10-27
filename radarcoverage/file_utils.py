# Utils
import os
import json


def json_save(filename, data, *args, indent=4, **kwargs):
    """
    Writes a dictionary into a json format file.

    :param filename: the filepath
    :type filename: str
    :param data: the dictionary
    :type data: dict
    :param args: positional arguments to be passed to :func:`json.dump`
    :type args: any
    :param indent: indentation size
    :type indent: int
    :param kwargs: keyword arguments to be passed to :func:`json.dump`
    """
    with open(filename, 'w') as f:
        json.dump(data, f, *args, indent=indent, **kwargs)


def json_load(filename, *args, **kwargs):
    """
    Read a dictionary from a json format file.

    :param filename: the filepath
    :type filename: str
    :param args: positional arguments to be passed to :func:`json.load`
    :type args: any
    :param kwargs: keyword arguments to be passed to :func:`json.load`
    :return: the dictionary
    :rtype: dict
    """
    with open(filename, 'r') as f:
        return json.load(f, *args, **kwargs)


def chdir_to_file_dir(filename):
    """
    Changes the working directory to be the same as the one containing filename.

    :param filename: the filename
    :type filename: str

    :Example:

    >>> os.getcwd()
    '/pathto/RadarCoverage'
    >>> chdir_to_file_dir(__file__)
    >>> os.getcwd()
    '/pathto/RadarCoverage/radarcoverage'
    """
    os.chdir(os.path.dirname(os.path.realpath(filename)))

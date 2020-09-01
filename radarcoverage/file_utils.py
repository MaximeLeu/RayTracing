# Utils
import os
import json


def json_save(filename, data):
    """
    Writes a dictionary into a json format file.

    :param filename: the filepath
    :type filename: str
    :param data: the dictionary
    :type data: dict
    """
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)


def json_load(filename):
    """
    Read a dictionary from a json format file.

    :param filename: the filepath
    :type filename: str
    :return: the dictionary
    :rtype: dict
    """
    with open(filename, 'r') as f:
        json.load(f)


def chdir_to_file_dir(filename):
    """
    Changes the working directory to be the same as the one containing filename.

    :param filename: the filename
    :type filename: str

    :Example:

    >>> os.getcwd()
    '/pathto/RadarCoverage'
    >>> chdir_to_file_dir(__file__)
    '/pathto/RadarCoverage/radarcoverage'
    """
    os.chdir(os.path.dirname(os.path.realpath(filename)))

import os


def chdir_to_file_dir(filename):
    os.chdir(os.path.dirname(os.path.realpath(filename)))

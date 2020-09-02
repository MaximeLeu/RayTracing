import numpy as np


def sort_by_columns(array, columns=None, axis=1):
    """
    Sorts an array by columns (or any other axis specified) in a given order.
    By default, will sort by columns, with first columns being the most decisive and the last column being the least
    one. This function's behavior is similar to the one of :func:`pandas.DataFrame.sort_values`.

    :param array: the array to be sorted
    :type array: numpy.ndarray
    :param columns: the indices to sort by
    :type columns: Iterable[int]
    :param axis: the axis to sort by
    :type axis: int
    :return: the sorted array
    :rtype: numpy.ndarray
    """
    if columns is None:
        columns = range(array.shape[axis])

    slices = list()

    for column in columns:
        slc = [slice(None)] * array.ndim
        slc[axis] = column
        slc = tuple(slc)
        slices.append(slc)

    arrays = [
        array[slc] for slc in slices
    ]

    r = np.core.records.fromarrays(arrays)

    return array[r.argsort()]

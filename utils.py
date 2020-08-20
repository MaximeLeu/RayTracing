import copy


def modify_class(obj, new_class, inplace=False):
    """
    Parses the object into given class.
    If not in place, will return a new object.

    :param obj: the object to be parsed
    :type obj: any
    :param new_class: the new class
    :type new_class: type
    :param inplace: if False, will return a copy
    :type inplace: bool
    :return: the object with new type class or nothing
    :rtype: new_class or None
    """
    if inplace:
        obj.__class__ = new_class
    else:
        obj_copy = copy.deepcopy(obj)
        obj_copy.__class__ = new_class
        return obj_copy

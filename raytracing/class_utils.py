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


def get_all_parent_classes(subclass):
    ret = list()
    for parent_class in subclass.__bases__:
        if parent_class != object:
            ret.extend(get_all_parent_classes(parent_class))
            ret.append(parent_class)
    return ret


def share_parent_class(class_1, class_2):
    parents_1 = set(get_all_parent_classes(class_1))
    parents_2 = set(get_all_parent_classes(class_2))
    return any(parent_1 in parents_2 for parent_1 in parents_1)

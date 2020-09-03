class ManyToOneDict(dict):
    """
    Subclass of dict providing multiple keys access to identical values.
    
    :Example:

    >>> d = ManyToOneDict()
    >>> d[5] = 4
    >>> d[4] = 1
    >>> d[1] = 0
    >>> d[4, 5, ...]  # Match any key containing 4 or 5
    [((5,), 4), ((4,), 1)
    >>> d[4, 5, ...] = 4
    >>> d[4, 5, ...]
    [((5,), 4), ((4,), 4)]
    >>> d[5, 1] = 0
    >>> d[5]  # Will always return a list of tuples
    [((5,), 4), ((5, 1), 0)]
    >>> d[5, 1]  # Will always return a tuple
    ((5, 1), 0)
    """
    def __setitem__(self, key, value):
        if not isinstance(key, tuple):
            key = (key,)
        elif key[-1] == Ellipsis:
            keys = key[:-1]
            for key in self.keys():
                if any(i in key for i in keys):
                    super().__setitem__(key, value)
        super().__setitem__(key, value)

    def __getitem__(self, item):
        if not isinstance(item, tuple):
            ret = []
            for key in self.keys():
                if item in key:
                    ret.append((key, super().__getitem__(key)))
            return ret
        elif item[-1] == Ellipsis:
            ret = []
            for key in self.keys():
                if any(i in key for i in item[:-1]):
                    ret.append((key, super().__getitem__(key)))
            return ret
        else:
            return item, super().__getitem__(item)

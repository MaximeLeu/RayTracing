class ManyToOneDict(dict):

    def __setitem__(self, key, value):
        if not isinstance(key, tuple):
            key = (key,)

        super().__setitem__(key, value)

    def __getitem__(self, item):
        if not isinstance(item, tuple):
            ret = []
            for key in self.keys():
                if item in key:
                    ret.append((key, super().__getitem__(key)))
            return ret
        else:
            return item, super().__getitem__(item)

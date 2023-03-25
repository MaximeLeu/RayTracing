import json
import numpy as np
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
        
        
    def __eq__(self, other):
        if not isinstance(other, ManyToOneDict):
            return False

        if len(self) != len(other):
            return False

        for key, value in self.items():
            if key not in other:
                return False
            other_value = super(ManyToOneDict, other).__getitem__(key)
            if isinstance(value, np.ndarray) and isinstance(other_value, np.ndarray):
                if not np.array_equal(value, other_value):
                    return False
            elif value != other_value:
                return False

        return True
    
    
    
    def to_json(self, filename=None):
        """
        Serialize the ManyToOneDict object to a JSON file.
        :param filename: The path to the JSON file where the object will be serialized.
        """
        serialized_data = {}
        for key, value in self.items():
            key_str = str(tuple([int(k) for k in key]))
            if isinstance(value, np.ndarray):
                value = value.tolist()
            serialized_data[key_str] = value
        
        if filename is not None:
            with open(filename, 'w') as json_file:
                json.dump(serialized_data, json_file)
        return json.dumps(serialized_data)
        
    @classmethod
    def from_json(cls, data=None, filename=None):
        """
        Deserialize a JSON file to a ManyToOneDict object.
        :param file_path: The path to the JSON file containing the serialized ManyToOneDict object.
        :return: A ManyToOneDict object containing the data from the JSON file.
        """
        if data is None and filename is not None:
            with open(filename, 'r') as json_file:
                data = json.load(json_file)

        if data is None:
            raise ValueError("Either data or filename must be provided")
        
        instance = cls()
        for key, value in data.items():
            # Convert string representation of tuples back to tuples
            key_tuple = tuple(int(x) for x in filter(None, key.strip("()").split(",")))
            if isinstance(value, list):
                value = np.array(value)
            instance[key_tuple] = value

        return instance

    
    
    
    
    
    
    
    
    
    
    
    
    
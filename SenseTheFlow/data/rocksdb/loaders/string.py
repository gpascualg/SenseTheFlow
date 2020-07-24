import numpy as np

from .bytes import RocksBytes


class RocksString(RocksBytes):
    def __init__(self, name, max_key_size=None, append=False, delete=False, read_only=False, dtype=np.float32, skip=None, num_samples=None):
        RocksBytes.__init__(self, name, max_key_size, append, delete, read_only, dtype, skip, num_samples)
    
    def put(self, data):
        return RocksBytes.put(self, data.encode())

    def iterate(self):
        for value in RocksBytes.iterate(self):
            yield value.decode()

    def split(self, num_samples):
        assert self.read_only, "Database must be in read only mode"

        split_a = RocksString(self.name, self.max_key_size, append=False, delete=False, read_only=True, dtype=self.dtype, skip=None, num_samples=num_samples)
        split_b = RocksString(self.name, self.max_key_size, append=False, delete=False, read_only=True, dtype=self.dtype, skip=num_samples+1, num_samples=None)
        return split_a, split_b

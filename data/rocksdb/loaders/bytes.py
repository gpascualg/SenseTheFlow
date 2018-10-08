import ctypes
import numpy as np

from ..db import RocksDB
from .wildcard import RocksWildcard


class RocksBytes(RocksWildcard):
    def __init__(self, name, max_key_size=None, append=False, delete=False, read_only=False, dtype=np.float32, skip=None, num_samples=None):
        # Initialize parent
        RocksWildcard.__init__(self, name, max_key_size, append, delete, read_only, dtype, skip, num_samples)

        # Open sub-databases
        self.db = RocksDB(name, self.max_key_size, read_only)

    
    def put(self, data, value_len=None):
        return self.put_ptr(ctypes.c_char_p(data), value_len or len(data))

    def put_ptr(self, ptr, value_len):
        key_str = RocksWildcard.get_key(self)
        return self.db.write(ctypes.c_char_p(key_str), ptr, key_len=self.max_key_size, value_len=value_len)
    
    def iterate(self):
        itr = self.db.iterator()
        self.itrs.append(itr)

        if self.skip is not None:
            for i in range(self.skip): itr.next()

        i = 0
        while itr.valid():
            ptr, plen = itr.value()
            label = (ctypes.c_char * plen).from_address(ptr)
            yield label.raw

            i += 1
            if self.num_samples is not None and i >= self.num_samples:
                break
            
            itr.next()

        self.itrs.remove(itr)
        itr.close()

    def split(self, num_samples):
        assert self.read_only, "Database must be in read only mode"

        split_a = RocksBytes(self.name, self.max_key_size, append=False, delete=False, read_only=True, dtype=self.dtype, skip=None, num_samples=num_samples)
        split_b = RocksBytes(self.name, self.max_key_size, append=False, delete=False, read_only=True, dtype=self.dtype, skip=num_samples+1, num_samples=None)
        return split_a, split_b

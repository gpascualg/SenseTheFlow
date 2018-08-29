import numpy as np
import ctypes

from ..db import RocksDB
from ..helper import serialize_numpy, unserialize_numpy
from .wildcard import RocksWildcard


class RocksNumpy(RocksWildcard):
    def __init__(self, name, max_key_size=None, append=False, delete=False, read_only=False, dtype=np.float32, skip=None, num_samples=None):
        # Initialize parent
        RocksWildcard.__init__(self, name, max_key_size, append, delete, read_only, dtype, skip, num_samples)

        # Open sub-databases
        self.db = RocksDB(name, self.max_key_size, read_only)

    def put(self, data):
        # We need flatten arrays
        if data.ndim > 1:
            # Save shape and flatten
            shape = data.shape
            data = data.ravel()

            # Not saved yet
            if self.iter_shape is None:
                self.iter_shape = shape
                self._save_metadata()
            # Check consistency
            else:
                if self.iter_shape != shape:
                    raise Exception("Expected constant shape")
        else:
            if self.iter_shape is not None:
                if np.prod(self.iter_shape) != np.prod(data.shape):
                    raise Exception("Expected constant shape")

        key_str = RocksWildcard.get_key(self)
        data, value_len, c = serialize_numpy(data, self.dtype)        
        return self.db.write(ctypes.c_char_p(key_str), data, key_len=self.max_key_size, value_len=value_len)

    def iterate(self):
        self.itr = itr = self.db.iterator()

        if self.skip is not None:
            for i in range(self.skip): itr.next()

        i = 0
        while self.itr is not None and itr.valid():
            ptr, plen = itr.value()
            value = unserialize_numpy((self.ctype * (plen // self.dsize)).from_address(ptr), self.dtype, self.iter_shape)
            yield value

            i += 1
            if self.num_samples is not None and i >= self.num_samples:
                break

            itr.next()

        itr.close()
        self.itr = None

    def split(self, num_samples):
        assert self.read_only, "Database must be in read only mode"

        split_a = RocksNumpy(self.name, self.max_key_size, append=False, delete=False, read_only=True, dtype=self.dtype, skip=None, num_samples=num_samples)
        split_b = RocksNumpy(self.name, self.max_key_size, append=False, delete=False, read_only=True, dtype=self.dtype, skip=num_samples+1, num_samples=None)
        return split_a, split_b


class RocksNonConstantNumpy(RocksNumpy):
    def __init__(self, name, max_key_size=None, append=False, delete=False, read_only=False, dtype=np.float32, skip=None, num_samples=None):
        # Initialize parent
        RocksNumpy.__init__(self, name, max_key_size, append, delete, read_only, dtype, skip, num_samples)

    def put(self, array):
        key_str = RocksWildcard.get_key(self)

        while array.ndim < 3:
            array = array[..., np.newaxis]

        assert array.ndim <= 3, "Only 3D Data supported"

        # Initialize buffer with the shape
        data, value_len, c = serialize_numpy(array, self.dtype)  
        shape_size = array.ndim * 4
        total_size = shape_size + value_len
        buffer = (ctypes.c_uint8 * total_size)(*list(array.shape))

        # Copy array contents
        ctypes.memmove(ctypes.addressof(buffer) + shape_size, data, value_len)

        # Cast to pointer
        buffer = ctypes.cast(buffer, ctypes.c_char_p)

        return self.db.write(ctypes.c_char_p(key_str), buffer, key_len=self.max_key_size, value_len=total_size)

    def iterate(self):
        self.itr = itr = self.db.iterator()

        if self.skip is not None:
            for i in range(self.skip): itr.next()

        i = 0
        while self.itr is not None and itr.valid():
            ptr, plen = itr.value()

            shape = list((ctypes.c_int * 3).from_address(ptr))
            value = unserialize_numpy((self.ctype * (plen // self.dsize)).from_address(ptr + 3 * 4), self.dtype, shape)
            yield value

            i += 1
            if self.num_samples is not None and i >= self.num_samples:
                break

            itr.next()

        itr.close()
        self.itr = None


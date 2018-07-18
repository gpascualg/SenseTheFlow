import os
import ctypes
import yaml
import shutil
import numpy as np

from ..helper import ROCKS_DB_POOL, types
from ..groups import store


class RocksWildcard(object):
    def __init__(self, name, max_key_size=None, append=False, delete=False, read_only=False, dtype=np.float32, skip=None, num_samples=None):
        if delete:
            try:
                shutil.rmtree(name)
            except:
                pass
            
        # Try to locate metada file
        try:
            # It might be the first run
            self.iter_shape = None

            with open(os.path.join(name, '.metadata')) as fp:
                self.metadata = yaml.load(fp)

                # Currently the only supported version
                if self.metadata['version'] == 1:
                    max_key_size = self.metadata['max_key_size']
                    dtype = self.metadata['dtype']
                    self.iter_shape = self.metadata['shape']
                else:
                    raise Exception("Unsupported metada version")

        except (yaml.YAMLError, IOError) as exc:
            if max_key_size is None:
                raise Exception("Expected non-None max_key_size")

        # Check given type
        self.ctype, self.dsize, self.dtype, self.typestr = types(dtype)
        
        # Base folder (try to create it)
        try:
            os.mkdir(name)
        except:
            pass

        # Attributes
        self.name = name
        self.last_key = 0
        self.max_key_size = max_key_size
        self.read_only = read_only
        self.itr = None
        self.db = None
        self.skip = skip
        self.num_samples = num_samples

        # Checkpoint metadata
        self._save_metadata()

        # Add to opened DBs pool
        ROCKS_DB_POOL.append(self)

        # If append, get last used key
        if append:
            raise NotImplementedError("Must be redone")

            # itr = self.values.iterator()
            # itr.last()
            
            # key_ptr, key_len = itr.key()
            # self.last_key = int((ctypes.c_char * key_len.value).from_address(key_ptr).value)

    def _save_metadata(self):
        # Save metadata
        with open(os.path.join(self.name, '.metadata'), 'w') as fp:
            yaml.dump({
                'version': 1,
                'max_key_size': self.max_key_size,
                'dtype': self.typestr,
                'shape': self.iter_shape
            }, fp)

    def get_key(self):
        self.last_key += 1
        return str(self.last_key).zfill(self.max_key_size).encode()

    def close_iterator(self):
        if self.itr is not None:
            self.itr.close()
            self.itr = None

    def close(self):
        self.close_iterator()

        if self.db is not None:
            self.db.close()
            self.db = None

    def __add__(self, other):
        if isinstance(other, RocksWildcard):
            store = RocksStore()
            store.add(self)
            store.add(other)
            return store
        
        if isinstance(other, store.RocksStore):
            other.insert(0, self)
            return other

        raise RuntimeError("Unexpected input")

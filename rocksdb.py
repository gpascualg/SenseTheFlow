import ctypes
import os
import shutil
import numpy as np
import yaml


##########################################
# Note: This will only work if running in a custom jupyter
#  environment that sends SIGTERM before restarting kernels,
#  otherwise jupyter sends SIGKILL directly, which is not
#  handlable
# Will also work on vanilla python environments (coda included)
import signal

ROCKS_DB_POOL = []

def signal_handler(signal, frame):
    global ROCKS_DB_POOL
    for db in ROCKS_DB_POOL:
        db.close()

signal.signal(signal.SIGTERM, signal_handler)
##########################################


class RocksConcat(object):
    def __init__(self, stores, elements_per_iter):
        self.stores = stores
        self.elements_per_iter = elements_per_iter

    def iterate(self, cyclic=True):
        has_done_epoch = [False] * len(self.stores)
        itrs = [store.iterate(cyclic=cyclic) for store in self.stores]

        while True:
            if all(has_done_epoch) and not cyclic:
                raise StopIteration()

            for pos in range(len(self.stores)):
                results = []
                itr = itrs[pos]

                while len(results) < self.elements_per_iter[pos]:
                    try:
                        results.append(next(itr))
                    except StopIteration:
                        itr = itrs[pos] = self.stores[pos].iterate(cyclic=cyclic)
                        has_done_epoch[pos] = True

                yield tuple(results)

    def concat(self, other, elements_per_iter):
        self.stores.append(other)
        self.elements_per_iter.append(elements_per_iter)


class RocksStore(object):
    def __init__(self):
        self.dbs = []

    def add(self, db):
        self.dbs.append(db)

    def split(self, num_samples):
        splits = [db.split(num_samples) for db in self.dbs]
        
        split_a = RocksStore()
        split_b = RocksStore()

        for db_a, db_b in splits:
            split_a.add(db_a)
            split_b.add(db_b)

        return split_a, split_b


    def concat(self, other, elements_per_iter_self=1, elements_per_iter_other=1):
        return RocksConcat([self, other], [elements_per_iter_self, elements_per_iter_other])

    def iterate(self):
        itrs = [db.iterate() for db in self.dbs]

        while True:
            yield tuple([next(itr) for itr in itrs])

    def close(self):
        for db in self.dbs:
            db.close()

    def close_iterator(self):
        for db in self.dbs:
            db.close_iterator()


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
                self.metadata = metadata = yaml.load(metadata)

                # Currently the only supported version
                if metadata['version'] == 1:
                    max_key_size = metadata['max_key_size']
                    dtype = metadata['dtype']
                    self.iter_shape = metadata['shape']
                else:
                    raise Exception("Unsupported metada version")

        except (yaml.YAMLError, IOError) as exc:
            if max_key_size is None:
                raise Exception("Expected non-None max_key_size")

        # Check given type
        if dtype == np.float32 or dtype == 'float' or dtype == float:
            self.ctype = ctypes.c_float
            self.dsize = 4
            self.dtype = np.float32
            self.typestr = 'float'
        elif dtype == np.float64 or dtype == 'double':
            self.ctype = ctypes.c_double
            self.dsize = 8
            self.dtype = np.float64
            self.typestr = 'double'
        elif dtype == np.uint8 or dtype == 'uint8':
            self.ctype = ctypes.c_uint8
            self.dsize = 1
            self.dtype = np.uint8
            self.typestr = 'uint8'
        else:
            raise Exception("Unkown data type")
        
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
        global ROCKS_DB_POOL
        ROCKS_DB_POOL.append(self)

        # If append, get last used key
        if append:
            raise NotImplementedError("Must be redone")

            itr = self.values.iterator()
            itr.last()
            
            key_ptr, key_len = itr.key()
            self.last_key = int((ctypes.c_char * key_len.value).from_address(key_ptr).value)

    def _save_metadata(self):
        # Save metadata
        with open(os.path.join(self.name, '.metadata'), 'w') as fp:
            yaml.dump({
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


class RocksNumpy(RocksWildcard):
    def __init__(self, name, max_key_size=None, append=False, delete=False, read_only=False, dtype=np.float32, skip=None, num_samples=None):
        # Initialize parent
        RocksWildcard.__init__(self, name, max_key_size, append, delete, read_only, dtype, skip, num_samples)

        # Open sub-databases
        self.db = RocksDB(name, max_key_size, read_only)

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
        contiguous = data.astype(self.dtype).copy(order='C')
        
        return self.db.write(ctypes.c_char_p(key_str), contiguous.ctypes.data_as(ctypes.c_char_p), 
                                   key_len=self.max_key_size, value_len=data.size * self.dsize)

    def iterate(self):
        # Must have a saved iter_shape
        if self.iter_shapeshape is None:
            raise Exception("A non-None shape was expected")

        self.itr = itr = self.db.iterator()

        if self.skip is not None:
            for i in range(self.skip): itr.next()

        i = 0
        while self.itr is not None and itr.valid():
            ptr, plen = itr.value()
            array_ptr = np.ctypeslib.as_array((self.ctype * (plen // self.dsize)).from_address(ptr))
            value = np.ctypeslib.as_array(array_ptr)
            yield value.reshape(self.iter_shape).astype(self.dtype)

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


class RocksBytes(RocksWildcard):
    def __init__(self, name, max_key_size=None, append=False, delete=False, read_only=False, dtype=np.float32, skip=None, num_samples=None):
        # Initialize parent
        RocksWildcard.__init__(self, name, max_key_size, append, delete, read_only, dtype, skip, num_samples)

        # Open sub-databases
        self.db = RocksDB(name, max_key_size, read_only)

    
    def put(self, data):
        assert isinstance(data, bytes)
        key_str = RocksWildcard.get_key(self)
        return self.db.write(ctypes.c_char_p(key_str), ctypes.c_char_p(data), key_len=self.max_key_size,
                                    value_len=len(data))
    
    def iterate(self):
        self.itr = itr = self.db.iterator()

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

        itr.close()
        self.itr = None

    def split(self, num_samples):
        assert self.read_only, "Database must be in read only mode"

        split_a = RocksBytes(self.name, self.max_key_size, append=False, delete=False, read_only=True, dtype=self.dtype, skip=None, num_samples=num_samples)
        split_b = RocksBytes(self.name, self.max_key_size, append=False, delete=False, read_only=True, dtype=self.dtype, skip=num_samples+1, num_samples=None)
        return split_a, split_b


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


class RocksIterator(object):
    def __init__(self, itr):
        self.itr = itr
        self.first()
        
    def first(self):
        RocksDLL.get().rocksdb_iter_seek_to_first(self.itr)
        
    def last(self):
        RocksDLL.get().rocksdb_iter_seek_to_last(self.itr)
        
    def __next__(self):
        RocksDLL.get().rocksdb_iter_next(self.itr)
    
    def next(self):
        RocksDLL.get().rocksdb_iter_next(self.itr)
        
    def valid(self):
        return RocksDLL.get().rocksdb_iter_valid(self.itr)
        
    def key(self):
        rlen = ctypes.c_size_t()
        ptr = RocksDLL.get().rocksdb_iter_key(self.itr, ctypes.pointer(rlen))
        key = (ctypes.c_char * rlen.value).from_address(ptr)
        return key.raw.decode(), rlen.value
        
    def value(self):
        rlen = ctypes.c_size_t()
        return RocksDLL.get().rocksdb_iter_value(self.itr, ctypes.pointer(rlen)), rlen.value
        
    def seek(self, key):
        key_ptr = ctypes.c_char_p(key.encode())
        RocksDLL.get().rocksdb_iter_seek(self.itr, key_ptr, len(key))

    def status(self):
        status = ctypes.c_char_p()
        status_ptr = ctypes.pointer(status)
        RocksDLL.get().rocksdb_iter_get_error(self.itr, status_ptr)
        return status.value

    def close(self):
        RocksDLL.get().rocksdb_iter_destroy(self.itr)
       
    
class RocksDB(object):    
    def __init__(self, name, max_key_size, read_only):
        dll = RocksDLL.get()
        
        #Options
        opts = dll.rocksdb_options_create()
        dll.rocksdb_options_set_create_if_missing(opts, 1)
        
        # Plain tables
        dll.rocksdb_options_set_allow_mmap_reads(opts, 1)
        
        slice_transform = dll.rocksdb_slicetransform_create_fixed_prefix(max_key_size)
        dll.rocksdb_options_set_prefix_extractor(opts, slice_transform)
        
        # policy = dll.rocksdb_filterpolicy_create_bloom(10);
        # dll.rocksdb_options_set_filter_policy(opts, policy);
        dll.rocksdb_options_set_plain_table_factory(opts, max_key_size, 10, 0.75, 16)
    
        # Disable compression
        dll.rocksdb_options_set_compression(opts, 0)
        dll.rocksdb_options_set_compression_options(opts, -14, -1, 0);
        
        compression_levels = (ctypes.c_int * 4)(*[0, 0, 0, 0])
        dll.rocksdb_options_set_compression_per_level(opts, compression_levels, 4);
        
        # Buffer, writing and so on
        dll.rocksdb_options_set_max_open_files(opts, 512)
        dll.rocksdb_options_set_write_buffer_size(opts, 512 * 1024 * 1024)
        dll.rocksdb_options_set_target_file_size_base(opts, 512 * 1024 * 1024)
        dll.rocksdb_options_set_max_write_buffer_number(opts, 4)
        dll.rocksdb_options_increase_parallelism(opts, 4)

        # Create
        self.db_err = ctypes.c_char_p()
        self.db_err_ptr = ctypes.pointer(self.db_err)
        if read_only:
            self.db = dll.rocksdb_open_for_read_only(opts, name.encode(), 0, self.db_err_ptr)
        else:
            self.db = dll.rocksdb_open(opts, name.encode(), self.db_err_ptr)
        self.check_error()
        
        # Create read/write opts
        self.read_opts = dll.rocksdb_readoptions_create()
        self.write_opts = dll.rocksdb_writeoptions_create()
        
    
    def check_error(self):
        if self.db_err.value is not None:
            raise Exception(self.db_err.value)
        
    
    def write(self, key, value, key_len=0, value_len=0):
        if key_len == 0: key_len = len(key)
        if value_len == 0: value_len = len(value)
        
        RocksDLL.get().rocksdb_put(self.db, self.write_opts, key, key_len, value, value_len, self.db_err_ptr)
        self.check_error()
    
        return True
    
    
    def read(self, key, key_len=0):
        if key_len == 0: key_len = len(key)
        
        rlen = ctypes.c_size_t()
        ptr = RocksDLL.get().rocksdb_get(self.db, self.read_opts, key.encode(), key_len, ctypes.pointer(rlen), self.db_err_ptr);
        self.check_error()
    
        return ptr, rlen
    
    
    def iterator(self):
        itr = RocksDLL.get().rocksdb_create_iterator(self.db, self.read_opts)
        return RocksIterator(itr)
        
    
    def close(self):
        RocksDLL.get().rocksdb_close(self.db)
        

class RocksDLL(object):
    rocksdll = None
    
    @staticmethod
    def get():
        if RocksDLL.rocksdll is None:
            RocksDLL.rocksdll = ctypes.cdll.LoadLibrary('librocksdb.so')

            # Some setup
            dll = RocksDLL.rocksdll

            dll.rocksdb_options_create.restype = ctypes.c_void_p

            dll.rocksdb_options_set_create_if_missing.argtypes = [ctypes.c_void_p, ctypes.c_uint8]
            dll.rocksdb_options_set_allow_mmap_reads.argtypes = [ctypes.c_void_p, ctypes.c_uint8]
            
            dll.rocksdb_slicetransform_create_fixed_prefix.restype = ctypes.c_void_p
            dll.rocksdb_slicetransform_create_fixed_prefix.argtypes = [ctypes.c_size_t]

            dll.rocksdb_options_set_prefix_extractor.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

            dll.rocksdb_options_set_plain_table_factory.argtypes = [ctypes.c_void_p, ctypes.c_uint32, 
                                                                    ctypes.c_int, ctypes.c_double,
                                                                    ctypes.c_size_t]

            dll.rocksdb_options_set_compression.argtypes = [ctypes.c_void_p, ctypes.c_int]
            dll.rocksdb_options_set_compression_options.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                                                    ctypes.c_int, ctypes.c_int]
            dll.rocksdb_options_set_compression_per_level.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int),
                                                                      ctypes.c_size_t]


            dll.rocksdb_options_set_max_open_files.argtypes = [ctypes.c_void_p, ctypes.c_int]
            dll.rocksdb_options_set_write_buffer_size.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
            dll.rocksdb_options_set_target_file_size_base.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
            dll.rocksdb_options_set_max_write_buffer_number.argtypes = [ctypes.c_void_p, ctypes.c_int]
            dll.rocksdb_options_increase_parallelism.argtypes = [ctypes.c_void_p, ctypes.c_int]

            dll.rocksdb_open.restype = ctypes.c_void_p
            dll.rocksdb_open.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_char_p)]

            dll.rocksdb_open_for_read_only.restype = ctypes.c_void_p
            dll.rocksdb_open_for_read_only.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint8, ctypes.POINTER(ctypes.c_char_p)]

            dll.rocksdb_readoptions_create.restype = ctypes.c_void_p
            dll.rocksdb_writeoptions_create.restype = ctypes.c_void_p
            dll.rocksdb_readoptions_set_total_order_seek.argtypes = [ctypes.c_void_p, ctypes.c_uint8]

            dll.rocksdb_put.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t,
                                        ctypes.c_char_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_char_p)]
            
            
            dll.rocksdb_get.restype = ctypes.c_char_p
            dll.rocksdb_get.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t,
                                        ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_char_p)]
            
            dll.rocksdb_create_iterator.restype = ctypes.c_void_p
            dll.rocksdb_create_iterator.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            
            dll.rocksdb_close.argtypes = [ctypes.c_void_p]
            
            dll.rocksdb_iter_seek_to_first.argtypes = [ctypes.c_void_p]
            dll.rocksdb_iter_seek_to_last.argtypes = [ctypes.c_void_p]
            dll.rocksdb_iter_next.argtypes = [ctypes.c_void_p]
            dll.rocksdb_iter_valid.restype = ctypes.c_uint8
            dll.rocksdb_iter_valid.argtypes = [ctypes.c_void_p]
            dll.rocksdb_iter_key.restype = ctypes.c_void_p
            dll.rocksdb_iter_key.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
            dll.rocksdb_iter_value.restype = ctypes.c_void_p
            dll.rocksdb_iter_value.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
            dll.rocksdb_iter_destroy.argtypes = [ctypes.c_void_p]
            dll.rocksdb_iter_seek.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t]
            dll.rocksdb_iter_get_error.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_char_p)]

        return RocksDLL.rocksdll


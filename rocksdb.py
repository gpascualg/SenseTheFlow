import ctypes
import os
import shutil
import numpy as np
from scipy import misc
import pandas as pd
import matplotlib.pyplot as plt
import tqdm as tq # conda install -c conda-forge tqdm
import traceback


class RocksStore(object):
    def __init__(self, name, max_key_size, append=False, delete=False, dtype=np.float32):
        if delete:
            try:
                shutil.rmtree(name)
            except:
                pass
            
        # Check given type
        if dtype == np.float32 or dtype == 'float' or dtype == float:
            self.ctype = ctypes.c_float
            self.dsize = 4
            self.dtype = np.float32
        elif dtype == np.float64 or dtype == 'double':
            self.ctype = ctypes.c_double
            self.dsize = 8
            self.dtype = np.float64
        elif dtype == np.uint8 or dtype == 'uint8':
            self.ctype = ctypes.c_uint8
            self.dsize = 1
            self.dtype = np.uint8
        else:
            raise Exception("Unkown data type")
            
        # Base folder (try to create it)
        try:
            os.mkdir(name)
        except:
            pass
        
        # Open sub-databases
        self.values = RocksDB(os.path.join(name, 'values.db'), max_key_size)
        self.labels = RocksDB(os.path.join(name, 'labels.db'), max_key_size)
        self.last_key = 0
        self.max_key_size = max_key_size
    
        # If append, get last used key
        if append:
            itr = self.values.iterator()
            itr.last()
            
            key_ptr, key_len = itr.key()
            self.last_key = int((ctypes.c_char * key_len.value).from_address(key_ptr).value)

    
    def put(self, label, data):
        self.last_key += 1
        
        key_str = str(self.last_key).zfill(self.max_key_size)
        contiguous = data.astype(self.dtype).copy(order='C')
        
        result = self.values.write(ctypes.c_char_p(key_str), contiguous.ctypes.data_as(ctypes.c_char_p), 
                                   key_len=self.max_key_size, value_len=data.size * self.dsize)
    
        label = str(label)
        result &= self.labels.write(key_str, ctypes.c_char_p(label), key_len=self.max_key_size,
                                    value_len=len(label))
        
        return result
    
    
    def iterate(self, shape, cyclic=True):
        itr_values = self.values.iterator()
        itr_labels = self.labels.iterator()
        size = np.prod(shape)
        n_cicles = 0
        
        while n_cicles == 0 or cyclic:
            n_cicles += 1
            
            while itr_values.valid() and itr_labels.valid():
                ptr_value, value_len = itr_values.value()
                ptr_label, label_len = itr_labels.value()

                array_ptr = np.ctypeslib.as_array((self.ctype * size).from_address(ptr_value))
                value = np.ctypeslib.as_array(array_ptr)
                label = (ctypes.c_char * label_len.value).from_address(ptr_label)
                yield value.reshape(shape).astype(self.dtype), str(label.value)

                itr_values.next()
                itr_labels.next()
                
            itr_values.first()
            itr_labels.first()
            
        itr_values.close()
        itr_labels.close()
        raise Exception("Iterator closed")
        
    
    def close(self):
        self.values.close()
        self.labels.close()
        
        
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
        return RocksDLL.get().rocksdb_iter_key(self.itr, ctypes.pointer(rlen)), rlen
        
    def value(self):
        rlen = ctypes.c_size_t()
        return RocksDLL.get().rocksdb_iter_value(self.itr, ctypes.pointer(rlen)), rlen
        
    def close(self):
        RocksDLL.get().rocksdb_iter_destroy(self.itr)
       
    
class RocksDB(object):    
    def __init__(self, name, max_key_size):
        dll = RocksDLL.get()
        
        #Options
        opts = dll.rocksdb_options_create()
        dll.rocksdb_options_set_create_if_missing(opts, 1)
        
        # Plain tables
        dll.rocksdb_options_set_allow_mmap_reads(opts, 1)
        
        slice_transform = dll.rocksdb_slicetransform_create_fixed_prefix(max_key_size)
        dll.rocksdb_options_set_prefix_extractor(opts, slice_transform)
        
#         policy = RocksDB.rocksdb.rocksdb_filterpolicy_create_bloom(10);
#         RocksDB.rocksdb.rocksdb_options_set_filter_policy(opts, policy);
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
        self.db = dll.rocksdb_open(opts, name, self.db_err_ptr)
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
        ptr = RocksDLL.get().rocksdb_get(self.db, self.read_opts, key, key_len, ctypes.byref(rlen), self.db_err_ptr);
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

            dll.rocksdb_readoptions_create.restype = ctypes.c_void_p
            dll.rocksdb_writeoptions_create.restype = ctypes.c_void_p

            dll.rocksdb_put.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t,
                                        ctypes.c_char_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_char_p)]
            
            
            dll.rocksdb_get.restype = ctypes.c_char_p
            dll.rocksdb_get.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t,
                                        ctypes.c_char_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_char_p)]
            
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
            
        return RocksDLL.rocksdll

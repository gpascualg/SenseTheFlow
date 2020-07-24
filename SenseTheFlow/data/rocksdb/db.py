import ctypes

from .dll import RocksDLL
from .iterator import RocksIterator


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
        dll.rocksdb_options_set_compression_options(opts, -14, -1, 0)
        
        compression_levels = (ctypes.c_int * 4)(*[0, 0, 0, 0])
        dll.rocksdb_options_set_compression_per_level(opts, compression_levels, 4)
        
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

from .dll import RocksDLL
import ctypes


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

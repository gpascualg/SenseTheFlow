import ctypes


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

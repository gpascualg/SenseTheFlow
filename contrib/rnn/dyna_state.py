import tensorflow as tf

from .utils import transpose_batch_time


class DynamicState(object):
    def __init__(self, max_time):
        self.max_time = max_time
        self.arrays = {}
        
    def create(self, name, dtype, clear_after_read=True, extra=1):
        self.arrays[name] = tf.TensorArray(dtype=dtype, size=self.max_time + extra, clear_after_read=clear_after_read)
        
    def write(self, name, time, value):
        self.arrays[name] = self.arrays[name].write(time, value)
    
    def read(self, name, time):
        return self.arrays[name].read(time)
        
    def unstack(self, name, value):
        self.arrays[name] = self.arrays[name].unstack(transpose_batch_time(value))
        
    def create_unstack(self, name, value, clear_after_read=True):
        self.create(name, dtype=value.dtype, clear_after_read=clear_after_read)
        self.unstack(name, value)
        
    def stack(self, name):
        return transpose_batch_time(self.arrays[name].stack())
    
    def encode(self):
        data = sorted(self.arrays.items(), key=lambda x: x[0])
        return tuple(x[1] for x in data)
    
    def decode(self, state):
        for i, key in enumerate(sorted(self.arrays.keys())):
            self.arrays[key] = state[i]

import numpy as np
import ctypes


##########################################
# Note: This will only work if running in a custom jupyter
#  environment that sends SIGTERM before restarting kernels,
#  otherwise jupyter sends SIGKILL directly, which is not
#  handable
# Will also work on vanilla python environments (conda included)
import signal

ROCKS_DB_POOL = []

def signal_handler(signal, frame):
    for db in ROCKS_DB_POOL:
        db.close()

signal.signal(signal.SIGTERM, signal_handler)
##########################################


def types(dtype):
    if dtype in (np.float32, 'float', 'float32', float):
        return ctypes.c_float, 4, np.float32, 'float'
    if dtype in (np.float64, 'float64', 'double'):
        return ctypes.c_double, 8, np.float64, 'double'
    elif dtype in (np.uint8, 'uint8'):
        return ctypes.c_uint8, 1, np.uint8, 'uint8'
    else:
        raise Exception("Unkown data type")

def serialize_numpy(data, dtype):
    _, dsize, dtype, _ = types(dtype)
    contiguous = data.astype(dtype).copy(order='C')
    return contiguous.ctypes.data_as(ctypes.c_char_p), data.size * dsize, contiguous

def unserialize_numpy(data, dtype, shape):
    _, _, dtype, _ = types(dtype)
    array_ptr = np.ctypeslib.as_array(data)
    value = np.ctypeslib.as_array(array_ptr).astype(dtype)

    if shape is None:
        return value
    
    return value.reshape(shape)


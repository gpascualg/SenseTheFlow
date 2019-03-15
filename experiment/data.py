from enum import Enum


class FetchMethod(Enum):
    # Copies the content to the host machine
    COPY = 0

    # Remote mounts the directory
    MOUNT = 1

class UriType(Enum):
    # Loads an absolute path from the NAS
    REMOTE_PATH = 0

    # Loads/Creates this experiment persistent path
    PERSISTENT = 1

    # Loads another experiment/model persistent path
    # but any changes won't be saved
    MODEL = 2

class DataType(object):
    def __init__(self, method, uri_type, uri):
        self.Method = method
        self.UriType = uri_type
        self.RemoteUri = uri
        self.LocalUri = None

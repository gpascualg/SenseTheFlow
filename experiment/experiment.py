import tempfile

from .data import DataType


class Experiment(object):
    Current = None
    
    def __init__(self, model, runable, on_data_ready=None, before_run=None, on_stop=None):
        assert Experiment.Current is None, "Only one experiment might be setup per session"
        Experiment.Current = self

        # Vars
        self.__model = model
        self.__data = []
        self.__gpu = None
        self.__temp_path = tempfile.TemporaryDirectory()
        self.__persistent_path = self.__temp_path.name

        # Callbacks
        self.__runable = runable
        self.__on_data_ready = on_data_ready
        self.__before_run = before_run
        self.__on_stop = on_stop

    def assign_gpu(self, gpu):
        self.__gpu = gpu

    def add_data(self, method, uri_type, uri=None):
        if uri_type == UriType.PERSISTENT:
            if any(x.UriType == uri_type for x in self.__data):
                raise AssertionError("Only one persistent storage per experiment is allowed")

            # We no longer need a temp path, use the provided one (on_data_ready sets it up)
            uri = self.__model.name()
            self.__temp_path = None
            self.__persistent_path = None
        
        self.__data.append(DataType(method, uri_type, uri))

    def get_data(self):
        return self.__data

    def on_data_ready(self, data):
        # Save persistent directory
        if data.UriType == UriType.PERSISTENT:
            self.__persistent_path = data.LocalUri
        
        if self.__on_data_ready:
            self.__on_data_ready(self.__model, data)

    def before_run(self):
        if self.__before_run:
            self.__before_run(self.__model)

    def run(self):
        self.__runable(self.__model)

    def stop(self):
        if self.__on_stop:
            self.__on_stop(self.__model)

    def get_gpu(self):
        assert self.__gpu is not None, "Got an invalid GPU"

        return self.__gpu

    def get_model_directory(self):
        return self.__persistent_path
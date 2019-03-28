import tempfile
import os

from .data import DataType, FetchMethod, UriType
from ..helper import DefaultNamespace, cmd_args
from .utils import discover


def _ask(message, valid):
    resp = None
    while resp is None:
        resp = input(message).lower()
        resp = resp if (resp in valid) else None
    return resp

class Experiment(object):
    Current = None
    
    def __init__(self, model, on_data_ready=None, before_run=None, on_stop=None):
        # Vars
        self.__model = model
        self.__data = []
        self.__gpu = None
        self.__temp_path = tempfile.TemporaryDirectory()
        self.__persistent_path = self.__temp_path.name
        self.__is_remote_execution = False

        # Discoverable
        self.__model_components = None
        self.__is_using_initialized_model = False

        # Callbacks
        self.__on_run = None
        self.__on_data_ready = on_data_ready
        self.__before_run = before_run
        self.__on_stop = on_stop

    def set_remote_execution(self):
        self.__is_remote_execution = True

    def assign_gpu(self, gpu):
        self.__gpu = gpu

    def add_data(self, method, uri_type, uri):
        if uri_type == UriType.PERSISTENT:
            if any(x.UriType == uri_type for x in self.__data):
                raise AssertionError("Only one persistent storage per experiment is allowed")

            # We no longer need a temp path, use the provided one (on_data_ready sets it up)            
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
        self.__on_run(self.__model)

    def stop(self):
        if self.__on_stop:
            self.__on_stop(self.__model)

    def get_gpu(self):
        assert self.__gpu is not None, "Got an invalid GPU"

        return self.__gpu

    def get_model_directory(self):
        if not self.__is_remote_execution:
            for data in self.__data:
                if data.UriType == UriType.PERSISTENT:
                    return data.RemoteUri

        return self.__persistent_path

    def run_local(self, callback, prepend_timestamp=False, append_timestamp=False, force_ascii_discover=False, delete_existing=False):
        model_dir = os.path.normpath(self.get_model_directory())
        model_name = os.path.basename(model_dir)
        model_dir = model_dir[:-len(model_name)]

        # Discover models
        return discover.discover(model_dir, model_name, lambda *args: self._continue_loading(callback, *args), 
                          prepend_timestamp, append_timestamp, delete_existing, force_ascii_discover)

    def run_remote(self, on_run):
        self.__on_run = on_run

    def _continue_loading(self, callback, model, is_using_initialized_model):
        self.__model_components = model
        self.__is_using_initialized_model = is_using_initialized_model
        callback()

    def assert_initialized(self):
        assert self.__is_using_initialized_model, "This model is not initialized"

    def __enter__(self):
        assert Experiment.Current is None, "Only one experiment might be setup per session"
        Experiment.Current = self

        # Probably return a subclass "ExperimentImpl" that encapsulates all
        # execution related methods
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        Experiment.Current = None


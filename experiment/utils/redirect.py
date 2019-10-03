import sys
import logging
import traceback as tb
import tensorflow as tf
from functools import wraps
from threading import Lock

from ...config import is_jupyter


def _get_caller(offset=2):
    from ..experiment import ExperimentRun, ExperimentHook
    
    f0 = sys._getframe(offset)
    
    experiment_run_code = ExperimentRun._run.__code__
    experiment_hook_code = ExperimentHook._call_callback.__code__
    
    # Search for a global experiment first
    f = f0.f_back
    while f:
        if f.f_code == experiment_run_code:
            return f.f_locals['self'].experiment
        if f.f_code == experiment_hook_code:
            return f.f_locals['experiment']
        f = f.f_back
        
    # Search for a forwarded function
    f = f0.f_back
    while f:
        code = f.f_code
        if code in GlobalOutput.Forwards.keys():
            return GlobalOutput.Forwards[code]
        f = f.f_back
    
    return None

class GlobalOutput(object):
    Instance = None
    Maps = {}
    Forwards = {}

    def __new__(cls, experiment):
        GlobalOutput.Instance = object.__new__(cls)
        return GlobalOutput.Instance

    def __init__(self, experiment):
        GlobalOutput.Maps[experiment] = self
        self.__out = None
        self.__experiment = experiment

    def create(self):
        if is_jupyter():
            import ipywidgets as widgets
            from IPython.display import display
            self.__out = widgets.Output()
            display(self.__out)

    def widget(self):
        return self.__out

    def capture(self, fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            with self:
                return fn(*args, **kwargs)
            
        return wrapper

    def __enter__(self):
        return Redirect().enter()

    def __exit__(self, exc_type, exc_val, exc_tb):
        return Redirect().exit(exc_type, exc_val, exc_tb)
            
    def forward(self, fn, *args, experiment=None, **kwargs):
        @GlobalOutput.Instance.capture
        def _impl():
            return fn(*args, **kwargs)
        
        GlobalOutput.Forwards[fn.__code__] = experiment or self.__experiment
        return _impl()

class Redirect(object):
    Instance = None
    
    def __new__(cls):
        if Redirect.Instance is None:
            Redirect.Instance = object.__new__(cls)
        return Redirect.Instance
    
    def __init__(self):
        self.__old_stdout = None
        self.__old_stderr = None
        self.__count = 0
        self.__lock = Lock()
    
    class Impl():
        def __init__(self, fn, stderr=False):
            self.fn = fn
            self.stderr = stderr

        def write(self, string, *args, **kwargs):
            caller = _get_caller()
            experiment = GlobalOutput.Maps.get(caller)
            if caller is not None and experiment is not None:
                if self.stderr:
                    experiment.widget().append_stderr(string)
                else:
                    experiment.widget().append_stdout(string)
            else:
                self.fn.write(string, *args, **kwargs)

        def flush(self, *args, **kwargs):
            caller = _get_caller()
            experiment = GlobalOutput.Maps.get(caller)
            if caller is None or experiment is None:
                return self.fn.flush(*args, **kwargs)
        
        def isatty(self):
            caller = _get_caller()
            experiment = GlobalOutput.Maps.get(caller)
            if caller is None or experiment is None:
                return self.fn.isatty()
            return False

        # IPython has methods to focus different outputs, fake them
        def __getattr__(self, attr):
            caller = _get_caller()
            experiment = GlobalOutput.Maps.get(caller)
            if caller is None or experiment is None:
                return getattr(self.fn, attr)
        
    def redirect_tf_logging(self):        
        # Hack tensorflow logging
        tf_logger = tf.get_logger()
        
        for handler in tf_logger.handlers[:]:
            tf_logger.removeHandler(handler)
        
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(logging.BASIC_FORMAT, None))
        tf_logger.addHandler(handler)
    
    def unredirect(self):
        if self.__old_stdout:
            sys.stdout = self.__old_stdout
            sys.stderr = self.__old_stderr
            self.__old_stdout = None
            self.redirect_tf_logging()

    def redirect(self):
        if self.__old_stdout is not None:
            return
        
        # Safe old
        self.__old_stdout = sys.stdout
        self.__old_stderr = sys.stderr
        
        # Map standard output
        sys.stdout = Redirect.Impl(self.__old_stdout)
        sys.stderr = Redirect.Impl(self.__old_stderr, True)
        self.redirect_tf_logging()

    def enter(self):
        with self.__lock:
            if self.__count == 0:
                self.redirect()
            self.__count += 1

    def exit(self, exc_type, exc_val, exc_tb):
        with self.__lock:
            self.__count -= 1
            
            # Print exception manually, otherwise we won't see it
            if exc_type is not None:
                tb.print_exc()
                    
            if self.__count == 0:                    
                self.unredirect()
             
            # TODO(gpascualg): Should we supress the exception?
            #return True

def capture_output(fn):
    def wrapper(*args, **kwargs):
        @GlobalOutput.Instance.capture
        def _impl():
            return fn(*args, **kwargs)
        
        return _impl()
    return wrapper

def forward(fn, *args, experiment=None, **kwargs):
    return GlobalOutput.Instance.forward(fn, *args, experiment=experiment, **kwargs)

## HACKY AREA - Make sure we capture dataset outputs

_from_generator = tf.data.Dataset.from_generator

@staticmethod
def patched_from_generator(generator, output_types, output_shapes=None, args=None):
    return _from_generator(
        generator=lambda: forward(generator),
        output_types=output_types,
        output_shapes=output_shapes,
        args=args
    )

tf.data.Dataset.from_generator = patched_from_generator


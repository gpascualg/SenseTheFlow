from ..model import Model as SyncModel
from .internals import Thread

import types
from queue import Queue


def _push_result(result, queue, callback):
    # Evaluate
    if queue is not None:
        queue.put(result)
    
    # Predict
    if callback is not None:
        callback(result)

def _wrap_execution(model, fnc, queue, callback, *args, **kwargs):
    call_results = fnc(*args, **kwargs)

    # Training produces no output (None)
    if isinstance(call_results, types.GeneratorType):
        for execution in call_results:
            if isinstance(execution, types.GeneratorType):
                # Each result is individually iterated when predicting
                for result in execution:
                    _push_result(result, queue, callback)
            else:
                # Execution is already a result, probably a dict from evaluate
                _push_result(execution, queue, callback)
    else:
        # TODO: Do we really want this? As of now, this will push None
        _push_result(call_results, queue, callback)

    model.clean()

class ExecutionWrapper(object):
    def __init__(self, model, fnc_name, fnc, *args, **kwargs):
        self.model = model

        self.__fnc_name = fnc_name
        self.__fnc = fnc
        self.__queue = None
        self.__thread = None
        self.__args = args
        self.__kwargs = kwargs

    def start(self, callback=None, force_results=False):
        assert not self.isRunning(), "Model is already running"

        # Maybe we need all results to get pushed
        if self.__fnc_name == 'evaluate' or force_results:
            self.__queue = Queue()

        self.__thread = Thread(target=_wrap_execution, args=(self.model, self.__fnc, self.__queue, callback) + self.__args, kwargs=self.__kwargs)

        # Attach to model instances and clean laters
        SyncModel.instances.append(self)
        self.model.clean_fnc(lambda: SyncModel.instances.remove(self))

        # Start running now
        self.__thread.start()

        # Give us back
        return self

    # Expose some functions
    def terminate(self):
        if self.isRunning():
            self.__thread.terminate()
            self.__thread.join()
            self.model.clean()
            return True

        return False

    def isRunning(self):
        return self.__thread is not None and self.__thread.isAlive()

    def wait(self):
        return self.__thread.join()

    def attach(self):
        # Right now it is a simply wrapper to redraw, might do something else later on
        self.model.redraw_bars()

    def results(self, block=False, timeout=0):
        if self.__queue is None:
            return None
        
        return self.__queue.get(block, timeout)

from ..model import Model as SyncModel
from .internals import Thread

from queue import Queue


def __wrap_train(self, model, fnc, queue, callback, *args, **kwargs):
    fnc(*args, **kwargs)
    model.clean()

def __wrap_iterable(self, model, fnc, queue, callback, *args, **kwargs):
    for generator in fnc(*args, **kwargs):
        for result in generator:
            # Evaluate
            if queue is not None:
                queue.put(result)
            
            # Predict
            if callback is not None:
                callback(result)

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

    def start(self, callback=None):
        assert not self.isRunning(), "Model is already running"

        self.__queue = Queue() if self.__fnc_name == 'evaluate' else None
        target = __wrap_train if self.__fnc_name == 'train' else __wrap_iterable
        self.__thread = Thread(target=target, args=(self.model, self.__fnc, self.__queue, callback) + self.__args, kwargs=self.__kwargs)

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
        return self.__thread.isAlive()

    def wait(self):
        return self.__thread.join()

    def attach(self):
        # Right now it is a simply wrapper to redraw, might do something else later on
        self.model.redraw_bars()

    def results(self, block=False, timeout=0):
        if self.__queue is None:
            return None
        
        return self.__queue.get(block, timeout)

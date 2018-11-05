from ..model import Model as SyncModel
from .model import Model
from .execution_wrapper import ExecutionWrapper
from .internals import Thread

import time


class ParallelTrainEval(object):
    def __init__(self, model, every_n_secs, train_parameters, eval_parameters):
        assert isinstance(model, Model), "Parallel evaluation can only be used with async models"
        assert isinstance(train_parameters, dict), "Train parameters must be a dictionary"
        assert isinstance(eval_parameters, dict), "Eval parameters must be a dictionary"
        
        self.model = model
        self.__thread = None
        self.__every_n_secs = every_n_secs
        self.__train_parameters = train_parameters
        self.__eval_parameters = eval_parameters

    def _wait_until_next(self):
        time.sleep(self.__every_n_secs)
        self.model.save()
        # We want this to lock! :D
        self.model._as_sync_model().evaluate(**self.__eval_parameters)

    def start(self):
        self.__thread = Thread(target=self._wait_until_next)
        self.__thread.start()
        # No locking here!
        self.__train_context = self.model.train(**self.__train_parameters)

    def terminate(self, force=False):
        if not force:
            # Wait for a nice stop (this might imply waiting for eval to end)
            # And then kill this thread
            self.__train_context.terminate()
            self.__thread.terminate()          
        else:
            # First kill eval, then train
            self.__thread.terminate()
            self.__train_context.terminate(True)

    def __getattribute__(self, name):
        try:
            attr = object.__getattribute__(self, name)
            return attr
        except:
            attr = ExecutionWrapper.__getattribute__(self.__train_context, name)
            return attr

        
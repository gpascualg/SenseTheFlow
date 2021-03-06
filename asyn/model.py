from ..model import Model as SyncModel
from .execution_wrapper import ExecutionWrapper
from .parallel_train_eval import ParallelTrainEval
from .tfhooks import AsyncTaskMode, AsyncTaskHook, create_async_task

import tensorflow as tf
import os


class Model(SyncModel):
    def __init__(self, *args, **kwargs):
        self.__model = SyncModel(*args, **kwargs)
        self.__async_task = AsyncTaskHook(1, self.__model)

        # Inject the hook
        self.__model._add_hook(self.__async_task)


    def __wrap(self, fnc_name, fnc, *args, **kwargs):
        wrapper = ExecutionWrapper(self, fnc_name, fnc, *args, **kwargs)

        # Predict requires a user submitted iter function
        if fnc_name == 'predict':
            return wrapper
            
        # Train/Eval is automatically started
        # TODO: Should we make this user configurable?
        return wrapper.start()

    def __getattribute__(self, name):
        try:
            attr = SyncModel.__getattribute__(self.__model, name)
            if name in ('train', 'predict', 'evaluate'):
                return lambda *args, **kwargs: self.__wrap(name, attr, *args, **kwargs)

            return attr
        except:
            pass

        attr = object.__getattribute__(self, name)
        return attr

    def __enter__(self):
        # Do not add now, we might end up running nothing
        # SyncModel.instances.append(self.__model)
        return self

    def __exit__(self, type, value, tb):
        # Do not clean now, it could (potentially) blow up everything
        # self.__model.clean()
        # SyncModel.current = None
        pass

    def __save_callback(self, model, run_context, step):
        # @tf.CheckpointSaverHook
        # Get saver from the SAVERS collection if present.
        collection_key = tf.GraphKeys.SAVERS
        savers = tf.get_collection(collection_key)
        if not savers:
            raise RuntimeError(
                "No items in collection {}. Please add a saver to the collection "
                "or provide a saver or scaffold.".format(collection_key))
                
        elif len(savers) > 1:
            raise RuntimeError(
                "More than one item in collection {}. "
                "Please indicate which one to use by passing it to the constructor.".
                format(collection_key))

        saver = savers[0]
        saver.save(run_context.session, os.path.join(self.__model.classifier().model_dir, 'model.ckpt'), global_step=step)

    def __stop_callback(self, model, run_context, step):
        run_context.request_stop()

    def _as_sync_model(self):
        return self.__model

    def train_eval(self, every_n_secs, train_parameters, eval_parameters):
        context = ParallelTrainEval(self, every_n_secs, train_parameters, eval_parameters)
        context.start()
        return context

    def save(self, block=True):
        task = create_async_task(self.__save_callback, AsyncTaskMode.AFTER_RUN)
        self.__async_task.push(task)

        if block:
            task.semaphore.acquire()

    def save_every(self, steps):
        task = create_async_task(self.__save_callback, AsyncTaskMode.AFTER_RUN, steps=steps, repetitive=True)
        self.__async_task.push(task)

    def stop(self, block=True):
        task = create_async_task(self.__stop_callback, AsyncTaskMode.AFTER_RUN)
        self.__model.stop_has_been_requested = True
        self.__async_task.push(task)

        if block:
            task.semaphore.acquire()
    

from ..model import Model as SyncModel
from .execution_wrapper import ExecutionWrapper
from .tfhooks import AsyncTaskMode, AsyncTaskHook, create_async_task
from .internals import Thread

import tensorflow as tf


class Model(object):
    def __init__(self, *args, **kwargs):
        self.__model = SyncModel(*args, **kwargs)
        self.__async_task = AsyncTaskHook(1, self.__model)

        # Inject the hook
        self.__model._add_hook(self.__async_task)

    def wrap(self, fnc, *args, **kwargs):
        def inner_wrap(model, fnc, *args, **kwargs):
            fnc(*args, **kwargs)
            model.clean()

        t = Thread(target=inner_wrap, args=(self.__model, fnc,) + args, kwargs=kwargs)
        return ExecutionWrapper(self, t)

    def __getattribute__(self, name):
        try:
            attr = object.__getattribute__(self, name)
            return attr
        except:
            pass
        
        attr = SyncModel.__getattribute__(self.__model, name)
        if name in ('train', 'test', 'evaluate'):
            return lambda *args, **kwargs: self.wrap(attr, *args, **kwargs)

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

    def __save_callback(self, model, session, step):
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
        saver.save(session, self.__model.classifier().model_dir, global_step=step)

    def save(self, block=True):
        task = create_async_task(self.__save_callback, AsyncTaskMode.AFTER_RUN)
        self.__async_task.push(task)

        if block:
            task.semaphore.acquire()

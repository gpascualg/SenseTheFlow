from ..model import Model as SyncModel
from .execution_wrapper import ExecutionWrapper
from .internals import Thread


class Model(object):
    def __init__(self, *args, **kwargs):
        self.model = SyncModel(*args, **kwargs)

    def wrap(self, fnc, *args, **kwargs):
        def inner_wrap(model, fnc, *args, **kwargs):
            fnc(*args, **kwargs)
            model.clean()

        t = Thread(target=inner_wrap, args=(self.model, fnc,) + args, kwargs=kwargs)
        return ExecutionWrapper(self.model, t)

    def __getattribute__(self, name):
        try:
            attr = object.__getattribute__(self, name)
            return attr
        except:
            pass
        
        attr = SyncModel.__getattribute__(self.model, name)
        if name in ('train', 'test', 'evaluate'):
            return lambda *args, **kwargs: self.wrap(attr, *args, **kwargs)

        return attr

    def __enter__(self):
        # Do not add now, we might end up running nothing
        # SyncModel.instances.append(self.model)
        return self

    def __exit__(self, type, value, tb):
        # Do not clean now, it could (potentially) blow up everything
        # self.model.clean()
        # SyncModel.current = None
        pass

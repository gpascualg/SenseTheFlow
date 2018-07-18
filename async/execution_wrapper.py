from ..model import Model as SyncModel


class ExecutionWrapper(object):
    def __init__(self, model, thread):
        self.model = model
        self.thread = thread

        # Attach to model instances and clean laters
        SyncModel.instances.append(self)
        self.model.clean_fnc(lambda: SyncModel.instances.remove(self))

        # Start running now
        self.thread.start()

    # Expose some functions
    def terminate(self):
        if self.isRunning():
            self.thread.terminate()
            self.thread.join()
            self.model.clean()
            return True

        return False

    def isRunning(self):
        return self.thread.isAlive()

    def attach(self):
        # Right now it is a simply wrapper to redraw, might do something else later on
        self.model.redraw_bars()

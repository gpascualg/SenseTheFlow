import tensorflow as tf
import enum
import heapq
from queue import Queue
from types import SimpleNamespace
from threading import Semaphore


class AsyncTaskMode(enum.Enum):
    BEGIN = 0
    AFTER_CREATE = 1
    BEFORE_RUN = 2
    AFTER_RUN = 3


def create_async_task(callback, mode, steps=1, repetitive=False):
    assert isinstance(callback, callable)
    assert isinstance(mode, AsyncTaskMode)
    assert isinstance(steps, int)

    # This tasks can not repeat
    assert mode != AsyncTaskMode.BEGIN or not repetitive
    assert mode != AsyncTaskMode.AFTER_CREATE or not repetitive

    # Non-repetitive tasks must be executed as soon as pused
    assert not repetitive or steps == 1

    task = SimpleNamespace(callback=callback, mode=mode, steps=steps, repetitive=repetitive, semaphore=Semaphore(0))
    return task
    

class AsyncTaskHook(tf.train.SessionRunHook):
    """Hook that requests stop at a specified step."""

    def __init__(self, num_steps, model):
        self._num_steps = num_steps
        self._queue = {k: Queue() for k in AsyncTaskMode}
        self._repetable = {k: [] for k in AsyncTaskMode}
        self._model = model
        self._global_step = -1

    def push(self, task):
        self._queue[task.mode].put(task, False)

    def _execute_task(self, task, session):
        task.callback(self._model, session, self._global_step)
        task.semaphore.release()

    def _execute(self, mode, session):
        # Move repetitive into lists (for faster iteration)
        # Execute non-reptitive
        # Iterate until `queue.get` returns None
        while True:
            task = self._queue[mode].get(False)
            if task is None:
                break

            if task.repetitive:
                self._repetable[mode].append(task)
            else:
                self._execute_task(task, session)

        # Iterate until either list is empty or task 
        for task in self._repetable[mode]:
            # Execute if its time
            if task.steps % self._global_step == 0:
                self._execute_task(task, session)

    def begin(self):
        self._global_step_tensor = tf.train.get_global_step()
        if self._global_step_tensor is None:
            raise RuntimeError("Global step should be created to use StopAtStepHook.")

        self._execute(AsyncTaskMode.BEGIN, None)

    def after_create_session(self, session, coord):
        self._execute(AsyncTaskMode.AFTER_CREATE, session)

    def before_run(self, run_context):  # pylint: disable=unused-argument
        self._execute(AsyncTaskMode.BEFORE_RUN, run_context.session)
        return tf.train.SessionRunArgs(self._global_step_tensor)

    def after_run(self, run_context, run_values):
        self._global_step = run_values.results + 1
        self._execute(AsyncTaskMode.AFTER_RUN, run_context.session)

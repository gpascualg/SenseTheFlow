from threading import Thread
from queue import Queue

from .tfhooks import EvalCallbackHook


class GeneratorFromEval(object):
    def __init__(self, model, interpreter, tensors):
        self.__eval_callback = EvalCallbackHook(
            step_callback=self._step,
            aggregate_callback=self._done,
            fetch_tensors=tensors
        )
        self.__model = model
        self.__interpreter = interpreter
        self.__queue = Queue()

        # Launch
        self.__thread = Thread(target=self._run)
        self.__thread.start()

    def _run(self):
        self.__model.evaluate(1, eval_callback=self.__eval_callback)

    def _step(self, model, results):
        result = self.__interpreter(model, results)
        self.__queue.put(result)

    def _done(self, mode, results):
        self.__queue.put(None)

    def generator(self):
        while True:
            data = self.__queue.get()
            if data is None:
                raise StopIteration
            yield data


import tensorflow as tf


class EvalCallbackHook(tf.compat.v1.train.SessionRunHook):
    def __init__(self, aggregate_callback=None, step_callback=None, fetch_tensors=None):
        self._aggregate_callback = aggregate_callback
        self._step_callback = step_callback
        self._fetch_tensors = fetch_tensors
        self._model = None
        self._step = 0
        self._k = 0

        self.tensors = ()
        self.names = ()

    def set_model(self, model):
        self._model = model

    def set_k(self, k):
        self._k = k

    def aggregate_callback(self, model, k, results):
        if self._aggregate_callback is not None:
            self._aggregate_callback(model, k, results)

    def begin(self):
        pass

    def after_create_session(self, session, coord):
        graph = tf.get_default_graph()
        tensors = [(graph.get_tensor_by_name(name + ":0"), name) for name in self._fetch_tensors]
        self.tensors, self.names = zip(*tensors)

    def before_run(self, run_context):
        if self._fetch_tensors is not None:
            return tf.compat.v1.train.SessionRunArgs(self.tensors)

    def after_run(self, run_context, run_values):
        if self._step_callback is not None:
            self._step += 1
            self._step_callback(self._model, dict(zip(self.names, run_values.results)), self._k, self._step)


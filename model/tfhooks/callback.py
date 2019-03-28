import tensorflow as tf


class CallbackHook(tf.compat.v1.train.SessionRunHook):
    """Hook that requests stop at a specified step."""

    def __init__(self, num_steps, callback, model):
        self._num_steps = num_steps
        self._callback = callback
        self._model = model

    def begin(self):
        self._global_step_tensor = tf.compat.v1.train.get_global_step()
        if self._global_step_tensor is None:
            raise RuntimeError("Global step should be created to use StopAtStepHook.")

    def after_create_session(self, session, coord):
        pass

    def before_run(self, run_context):  # pylint: disable=unused-argument
        return tf.compat.v1.train.SessionRunArgs(self._global_step_tensor)

    def after_run(self, run_context, run_values):
        global_step = run_values.results + 1
        if global_step % self._num_steps == 0:
            self._callback(self._model, global_step)

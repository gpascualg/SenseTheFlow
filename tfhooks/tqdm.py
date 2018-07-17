import tensorflow as tf


class TqdmHook(tf.train.SessionRunHook):
    def __init__(self, model):
        self._model = model
        self._last_step = 0

    def begin(self):
        self._global_step_tensor = tf.train.get_global_step()
        if self._global_step_tensor is None:
            raise RuntimeError("Global step should be created to use StopAtStepHook.")

    def after_create_session(self, session, coord):
        pass

    def before_run(self, run_context):  # pylint: disable=unused-argument
        loss = [x for x in tf.get_collection(tf.GraphKeys.LOSSES)]
        return tf.train.SessionRunArgs(loss + [self._global_step_tensor])

    def after_run(self, run_context, run_values):
        loss, global_step = sum(run_values.results[:-1]), run_values.results[-1]
        update = global_step - self._last_step
        self._last_step = global_step

        _, bar = self._model.bar()
        bar.update(update)
        bar.set_description('Loss: {}'.format(loss))
   

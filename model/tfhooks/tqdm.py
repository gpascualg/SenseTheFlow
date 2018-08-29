import tensorflow as tf
import tqdm

from ...config import bar


class TqdmWrapper(object):
    def __init__(self, epochs=0, leave=True):
        # Private
        self.__epochs = epochs
        self.__leave = leave
        self.__hook = None

        # Exposed
        self.epoch = 0
        self.epoch_bar = None
        self.step_bar = None

    def create(self):
        self.__hook = TqdmHook(self)
        return self.__hook

    def draw(self):
        # Epochs bar is not shown in text-environments
        if bar != tqdm.tqdm:
            self.epoch_bar = bar(total=self.__epochs, leave=self.__leave)
            self.epoch_bar.update(self.epoch)
        
        # Create steps bar
        self.step_bar = bar(leave=self.__leave)

        # If it is not the first time (ie. already running)
        if self.__hook is not None:
            self.__hook.force_update()

    def update_epoch(self, epoch):
        if self.epoch_bar is not None:
            self.epoch_bar.update(epoch - self.epoch)
            self.epoch = epoch

    def done(self, force=False):
        if self.epoch_bar is not None:
            self.epoch_bar.close()
        
        if self.step_bar is not None:
            self.step_bar.close()


class TqdmHook(tf.train.SessionRunHook):
    def __init__(self, wrapper):
        self._wrapper = wrapper
        self._last_step = 0
        self._forced_update = False

    def force_update(self):
        self._forced_update = True

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

        if self._forced_update:
            self._forced_update = False
            update = global_step

        bar = self._wrapper.step_bar
        bar.update(update)
        bar.set_description('Loss: {}'.format(loss))
   

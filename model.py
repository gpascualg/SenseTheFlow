import types
import tensorflow as tf
import numpy as np
from .config import bar
from heapq import heappush, heappop
from functools import partial
import os

try:
    from inspect import signature
except:
    from funcsigs import signature


class CallbackHook(tf.train.SessionRunHook):
    """Hook that requests stop at a specified step."""

    def __init__(self, num_steps, callback, model):
        self._num_steps = num_steps
        self._callback = callback
        self._model = model

    def begin(self):
        self._global_step_tensor = tf.train.get_global_step()
        if self._global_step_tensor is None:
            raise RuntimeError("Global step should be created to use StopAtStepHook.")

    def after_create_session(self, session, coord):
        pass

    def before_run(self, run_context):  # pylint: disable=unused-argument
        return tf.train.SessionRunArgs(self._global_step_tensor)

    def after_run(self, run_context, run_values):
        global_step = run_values.results
        if (global_step - 2) % self._num_steps == 0:
            self._callback(self._model, global_step)


class TqdmHook(tf.train.SessionRunHook):
    def __init__(self, bar):
        self._bar = bar
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
        loss, global_step = run_values.results[0], run_values.results[-1]
        update = global_step - self._last_step
        self._last_step = global_step

        self._bar.update(update)
        self._bar.set_description('Loss: {}'.format(loss))


class CustomSummarySaverHook(tf.train.SummarySaverHook):
    def __init__(self,
               save_steps=None,
               save_secs=None,
               output_dir=None,
               summary_writer=None,
               scaffold=None,
               summary_op=None):

        tf.train.SummarySaverHook.__init__(self, 
            save_steps=save_steps,
            save_secs=save_secs,
            output_dir=output_dir,
            summary_writer=summary_writer,
            scaffold=scaffold,
            summary_op=summary_op)

    def _get_summary_op(self):
        tensors = [x for x in tf.get_collection(tf.GraphKeys.SUMMARIES) if x.op.name in self._summary_op]

        if len(tensors) != len(self._summary_op):
            tf.logging.error('Some tensors where not found')
            tnames = [x.op.name for x in tensors]
            tf.logging.error(set(self._summary_op) - set(tnames))

        return tensors

class DataParser(object):
    def __init__(self):
        self.__input_fn = dict(
            (tf.estimator.ModeKeys.TRAIN, None)
            (tf.estimator.ModeKeys.PREDICT, None)
            (tf.estimator.ModeKeys.EVAL, None)
        )

    def from_generator(self, generator, output_types, output_shapes=None, 
        pre_shuffle=False, post_shuffle=False, flatten=False, 
        num_samples=None, batch_size=1,
        mode=tf.estimator.ModeKeys.TRAIN):

        generator = {
            'generator': generator, 
            'output_types': output_types,
            'output_shapes': output_shapes
        }

        self.__input_fn[mode] = lambda num_epochs: self.generator_input_fn(
            generator, 
            pre_shuffle=pre_shuffle, post_shuffle=post_shuffle, flatten=flatten, 
            num_samples=num_samples, batch_size=batch_size,
            mode=mode, num_epochs=num_epochs
        )
        
        return self

    def train_from_generator(self, *args, **kwargs):
        self.from_generator(*args, mode=tf.estimator.ModeKeys.TRAIN, **kwargs)
        return self
    
    def eval_from_generator(self, *args, **kwargs):
        self.from_generator(*args, mode=tf.estimator.ModeKeys.EVAL, **kwargs)
        return self
        
    def predict_from_generator(self, *args, **kwargs):
        self.from_generator(*args, mode=tf.estimator.ModeKeys.PREDICT, **kwargs)
        return self

    def generator_input_fn(self, generator, parser_fn, mode, 
        pre_shuffle=False, post_shuffle=False, flatten=False, 
        num_samples=None, batch_size=1, num_epochs=1):

        dataset = tf.data.Dataset.from_generator(**generator)

        # Pre-parsing shuffle
        if pre_shuffle:
            dataset = dataset.shuffle(buffer_size=pre_shuffle)

        dataset = dataset.map(lambda *args: parser_fn(*args, mode=mode), num_parallel_calls=5)

        if flatten:
            dataset = dataset.flat_map(lambda x, y: tf.data.Dataset.from_tensor_slices((x, y)))

        if batch_size > 0:
            dataset = dataset.prefetch(batch_size)

        # Post-parsing
        if post_shuffle:
            dataset = dataset.shuffle(buffer_size=post_shuffle)

        if num_samples is not None:
            dataset = dataset.take(num_samples)

        dataset = dataset.repeat(num_epochs)
        if batch_size > 0:
            dataset = dataset.batch(batch_size)
            
        iterator = dataset.make_one_shot_iterator()
        features, labels = iterator.get_next()
        return features, labels

    def input_fn(self, mode, num_epochs):
        assert self.__input_fn[mode] is not None
        return self.__input_fn[mode](num_epochs)

class Model(object):
    current = None
    
    def __init__(self, data_parser, model_fn, model_dir, 
        config=None, run_config=None, warm_start_from=None, params={}):

        os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

        self.__config = config
        self.__data_parser = data_parser
        
        self.__epoch = 0
        self.__init = None
        self.__feed_dict = None
        self.__last_feed_dict = None
        self.__results = None
        self.__epoch_bar = None
        self.__step_bar = None
        
        self.__classifier = None
        self.__callbacks = []

        # Set up a RunConfig to only save checkpoints once per training cycle.
        if run_config is None:
            run_config = tf.estimator.RunConfig() \
                .replace(save_checkpoints_secs=1e9)

        run_config = run_config \
            .replace(session_config=self.__config)

        self.__classifier = tf.estimator.Estimator(
            model_fn=model_fn, model_dir=model_dir, config=run_config,
            warm_start_from=warm_start_from, params=params)
        
    def add_callback(self, steps, func):
        assert type(func) == types.FunctionType, "Expected a function in func"
        assert len(signature(func).parameters) == 2, "Expected func to have 2 parameters (model, step)"
        
        self.__callbacks.append(CallbackHook(steps, func, self))

    # Getters
    def epoch(self):
        return self.__epoch
    
    def bar(self):
        return self.__epoch_bar, self.__step_bar
    
    def classifier(self):
        return self.__classifier

    def __enter__(self):
        Model.current = self
        return self

    def __exit__(self, type, value, tb):
        Model.current = None       


    def _estimator_hook(self, func, mode, steps, callback, tf_hooks, log, summary):
        def hook(hooks, model, step):
            results = func(
                input_fn=lambda: self.__data_parser.input_fn(mode=mode, num_epochs=epochs_per_eval),
                hooks=hooks
            )
            callback(results)

        if log:
            tf_hooks = tf_hooks or []
            tf_hooks.append(tf.train.LoggingTensorHook(
                tensors=log,
                every_n_iter=1
            ))

        if summary:
            tf_hooks = tf_hooks or []

            tf_hooks.append(CustomSummarySaverHook(
                summary_op=summary,
                save_steps=1,
                output_dir=os.path.join(self.classifier().model_dir, func[:4])
            ))

        self.__callbacks.append(tf.train.CheckpointSaverHook(
            checkpoint_dir=self.classifier().model_dir,
            save_steps=steps
        ))

        self.add_callback(steps, lambda model, step: hook(tf_hooks, model, step))

    def eval_hook(self, steps, callback, tf_hooks=None, log=None, summary=None):
        self._estimator_hook(mode.classifier().evaluate, tf.estimator.ModeKeys.EVAL, steps, callback, tf_hooks, log, summary)

    def predict_hook(self, steps, callback, tf_hooks=None, log=None, summary=None):
        self._estimator_hook(mode.classifier().predict, tf.estimator.ModeKeys.PREDICT, steps, callback, tf_hooks, log, summary)
    
    def train(self, epochs, epochs_per_eval):
        self.__epoch_bar = bar(total=epochs)
        self.__step_bar = bar()
        self.__callbacks += [TqdmHook(self.__step_bar)]
        for self.__epoch in range(0, epochs, epochs_per_eval):
            logger = tf.train.LoggingTensorHook(
                tensors={
                    'global_step/step': 'global_step'
                },
                every_n_iter=10
            )

            step_counter = tf.train.StepCounterHook(every_n_steps=10, output_dir=self.classifier().model_dir)

            self.classifier().train(
                input_fn=lambda: self.__data_parser.input_fn(mode=tf.estimator.ModeKeys.TRAIN, num_epochs=epochs_per_eval),
                hooks=self.__callbacks + [logger, step_counter]
            )

            self.__epoch_bar.update(epochs_per_eval)

    def predict(self, epochs):
        self.__step_bar = bar()

        return self.classifier().predict(
            input_fn=lambda: self.__data_parser.input_fn(mode=tf.estimator.ModeKeys.PREDICT, num_epochs=epochs),
            hooks=[TqdmHook(self.__step_bar)]
        )

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
        global_step = run_values.results + 1
        if global_step % self._num_steps == 0:
            self._callback(self._model, global_step)


class TqdmHook(tf.train.SessionRunHook):
    def __init__(self, bar, estimator):
        self._bar = bar
        self._estimator = estimator
        self._last_step = 0

    def begin(self):
        self._global_step_tensor = tf.train.get_global_step()
        if self._global_step_tensor is None:
            raise RuntimeError("Global step should be created to use StopAtStepHook.")

    def after_create_session(self, session, coord):
        pass

    def before_run(self, run_context):  # pylint: disable=unused-argument
        loss = [x for x in tf.get_collection(tf.GraphKeys.SUMMARIES) if x.op.name == 'loss']
        return tf.train.SessionRunArgs(loss + [self._global_step_tensor])
        # return tf.train.SessionRunArgs([
        #     self._global_step_tensor,
        #     loss # HOW TO GET LOSS??
        # ])

    def after_run(self, run_context, run_values):
        print(run_values)
        print(run_values.results)
        global_step = run_values.results + 1
        update = global_step - self._last_step
        self._last_step = global_step

        self._bar.update(update)


class Model(object):
    current = None
    
    def __init__(self, parser_fn, model_fn, generator, batch_size, 
                pre_shuffle=False, post_shuffle=True, flatten=False, config=None):
        self.__config = config
        self.__parser_fn = parser_fn
        self.__model_fn = model_fn

        self.__data_generator = generator
        self.__batch_size = batch_size
        self.__pre_shuffle = pre_shuffle
        self.__post_shuffle = post_shuffle
        self.__flatten = flatten
        
        self.__epoch = 0
        self.__init = None
        self.__feed_dict = None
        self.__last_feed_dict = None
        self.__results = None
        self.__epoch_bar = None
        self.__step_bar = None
        
        self.__callbacks = []
        
    def add_callback(self, steps, func):
        assert type(func) == types.FunctionType, "Expected a function in func"
        assert len(signature(func).parameters) == 2, "Expected func to have 2 parameters (model, step)"
        
        self.__callbacks.append(CallbackHook(steps, func, self))

    def input_fn(self, is_training, shuffle_buffer=64, num_parallel_calls=5, num_epochs=1):
        if 'train' in self.__data_generator:
            if is_training:
                data_generator = self.__data_generator['train'] 
            else:
                data_generator = self.__data_generator['test'] 
        else:
            data_generator = self.__data_generator

        dataset = tf.data.Dataset.from_generator(**self.__data_generator)

        # Pre-parsing
        if is_training and self.__pre_shuffle:
            dataset.shuffle(buffer_size=shuffle_buffer)

        dataset = dataset.map(lambda features, labels: self.__parser_fn(features, labels, is_training), num_parallel_calls=num_parallel_calls)

        if self.__flatten:
            dataset = dataset.flat_map(lambda x, y: tf.data.Dataset.from_tensor_slices((x, y)))

        dataset = dataset.prefetch(self.__batch_size)

        # Post-parsing
        if is_training and self.__post_shuffle:
            dataset.shuffle(buffer_size=shuffle_buffer)

        dataset = dataset.repeat(num_epochs)
        if self.__batch_size > 0:
            dataset = dataset.batch(self.__batch_size)
            
        iterator = dataset.make_one_shot_iterator()
        features, labels = iterator.get_next()
        return features, labels

    # Getters
    def epoch(self):
        return self.__epoch
    
    def bar(self):
        return self.__epoch_bar, self.__step_bar
    
        
    def __enter__(self):
        Model.current = self
        return self

    def __exit__(self, type, value, tb):
        Model.current = None       


    def train(self, model_dir, epochs, epochs_per_eval):
        os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

        # Set up a RunConfig to only save checkpoints once per training cycle.
        run_config = tf.estimator.RunConfig() \
            .replace(save_checkpoints_secs=1e9) \
            .replace(session_config=self.__config)

        classifier = tf.estimator.Estimator(
            model_fn=self.__model_fn, model_dir=model_dir, config=run_config,
            params={})

        self.__epoch_bar = bar(range(epochs // epochs_per_eval))
        self.__step_bar = bar()
        self.__callbacks += [TqdmHook(self.__step_bar, classifier)]
        for self.__epoch in self.__epoch_bar:
            logger = tf.train.LoggingTensorHook(
                tensors={
                    'global_step/step': 'global_step'
                },
                every_n_iter=10
            )

            step_counter = tf.train.StepCounterHook(every_n_steps=10, output_dir=model_dir)

            classifier.train(
                input_fn=lambda: self.input_fn(True, shuffle_buffer=64, num_parallel_calls=5, num_epochs=epochs_per_eval),
                hooks=self.__callbacks + [logger, step_counter]
            )
        

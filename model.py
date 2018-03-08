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

class Model(object):
    current = None
    
    def __init__(self, parser_fn, model_fn, model_dir, batch_size, 
                shuffle_test=False, pre_shuffle=False, post_shuffle=False, 
                flatten=False, config=None, test_amount=5, 
                run_config=None, warm_start_from=None, params={}):

        os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

        self.__config = config
        self.__parser_fn = parser_fn
        
        self.__input_fn = None
        self.__data_generator = None

        self.__batch_size = batch_size
        self.__shuffle_test = shuffle_test
        self.__pre_shuffle = pre_shuffle
        self.__post_shuffle = post_shuffle
        self.__flatten = flatten
        self.__test_amount = test_amount
        
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

    def from_generator(self, generator, output_types, output_shapes=None):
        self.__data_generator = {
               'generator': generator, 
               'output_types': output_types,
               'output_shapes': output_shapes
           }
        self.__input_fn = self.generator_input_fn
        
    def add_callback(self, steps, func):
        assert type(func) == types.FunctionType, "Expected a function in func"
        assert len(signature(func).parameters) == 2, "Expected func to have 2 parameters (model, step)"
        
        self.__callbacks.append(CallbackHook(steps, func, self))

    def generator_input_fn(self, is_training, num_parallel_calls=5, num_epochs=1):
        if 'train' in self.__data_generator:
            if is_training:
                data_generator = self.__data_generator['train'] 
            else:
                data_generator = self.__data_generator['test'] 
        else:
            data_generator = self.__data_generator

        dataset = tf.data.Dataset.from_generator(**data_generator)

        # Pre-parsing
        if (is_training or self.__shuffle_test) and self.__pre_shuffle:
            dataset = dataset.shuffle(buffer_size=self.__pre_shuffle)

        dataset = dataset.map(lambda *args: self.__parser_fn(*args, is_training=is_training), num_parallel_calls=num_parallel_calls)

        if self.__flatten:
            dataset = dataset.flat_map(lambda x, y: tf.data.Dataset.from_tensor_slices((x, y)))

        dataset = dataset.prefetch(self.__batch_size)

        # Post-parsing
        if (is_training or self.__shuffle_test) and self.__post_shuffle:
            dataset = dataset.shuffle(buffer_size=self.__post_shuffle)

        if not is_training:
            dataset = dataset.take(self.__test_amount)

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
    
    def classifier(self):
        return self.__classifier

    def __enter__(self):
        Model.current = self
        return self

    def __exit__(self, type, value, tb):
        Model.current = None       


    def _estimator_hook(self, func, steps, callback, tf_hooks, log, summary):
        def hook(hooks, model, step):
            results = getattr(model.classifier(), func)(
                input_fn=lambda: self.__input_fn(False, num_parallel_calls=5, num_epochs=1),
                #checkpoint_path=os.path.join(model.classifier().model_dir, 'model.ckpt'),
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
        self._estimator_hook('evaluate', steps, callback, tf_hooks, log, summary)

    def predict_hook(self, steps, callback, tf_hooks=None, log=None, summary=None):
        self._estimator_hook('predict', steps, callback, tf_hooks, log, summary)
    
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
                input_fn=lambda: self.__input_fn(True, num_parallel_calls=5, num_epochs=epochs_per_eval),
                hooks=self.__callbacks + [logger, step_counter]
            )

            self.__epoch_bar.update(epochs_per_eval)


    def predict(self, epochs):
        self.__step_bar = bar()

        return self.classifier().predict(
            input_fn=lambda: self.__input_fn(False, num_parallel_calls=5, num_epochs=epochs),
            hooks=[TqdmHook(self.__step_bar)]
        )


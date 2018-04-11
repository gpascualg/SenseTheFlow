import types
import tensorflow as tf
from tensorflow.python import debug as tf_debug
import numpy as np
from .config import bar
from heapq import heappush, heappop
from functools import partial
from queue import Queue
from threading import Thread
from types import SimpleNamespace
import shutil
import argparse
import os

try:
    from inspect import signature
except:
    from funcsigs import signature


# Load some arguments from console
try:
    get_ipython()
    args = SimpleNamespace(debug=False)
except:
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', default=False, action='store_true')
    args = parser.parse_args()


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
        loss, global_step = sum(run_values.results[:-1]), run_values.results[-1]
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
        self.__input_fn = dict([
            (tf.estimator.ModeKeys.TRAIN, None),
            (tf.estimator.ModeKeys.PREDICT, None),
            (tf.estimator.ModeKeys.EVAL, None)
        ])

    def has(self, mode):
        return self.__input_fn[mode] is not None

    def from_generator(self, parser_fn, generator, output_types, output_shapes=None, 
        pre_shuffle=False, post_shuffle=False, flatten=False, 
        num_samples=None, batch_size=1,
        mode=tf.estimator.ModeKeys.TRAIN):

        generator = {
            'generator': generator, 
            'output_types': output_types,
            'output_shapes': output_shapes
        }

        self.__input_fn[mode] = lambda num_epochs: self.generator_input_fn(
            generator,  parser_fn=parser_fn,
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
        skip=None, num_samples=None, batch_size=1, num_epochs=1):

        dataset = tf.data.Dataset.from_generator(**generator)

        # Pre-parsing shuffle
        if pre_shuffle:
            dataset = dataset.shuffle(buffer_size=pre_shuffle)

        if skip is not None:
            dataset = dataset.skip(skip)

        dataset = dataset.map(lambda *args: parser_fn(*args, mode=mode), num_parallel_calls=5)

        if flatten:
            dataset = dataset.flat_map(lambda *args: tf.data.Dataset.from_tensor_slices((*args,)))

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

        if mode == tf.estimator.ModeKeys.PREDICT:
            features = iterator.get_next()
            return features
        
        features, labels = iterator.get_next()
        return features, labels

    def input_fn(self, mode, num_epochs):
        assert self.__input_fn[mode] is not None
        return self.__input_fn[mode](num_epochs)

class EvalCallback(tf.train.SessionRunHook):
    def __init__(self, aggregate_callback=None, step_callback=None, fetch_tensors=None):
        self._aggregate_callback = aggregate_callback
        self._step_callback = step_callback
        self._fetch_tensors = fetch_tensors
        self._model = None
        self._step = 0

        self.tensors = ()
        self.names = ()

    def set_model(self, model):
        self._model = model

    def aggregate_callback(self, results):
        if self._aggregate_callback is not None:
            self._aggregate_callback(self._model, results)

    def begin(self):
        pass

    def after_create_session(self, session, coord):
        graph = tf.get_default_graph()
        tensors = [(graph.get_tensor_by_name(name + ":0"), name) for name in self._fetch_tensors]
        self.tensors, self.names = zip(*tensors)

    def before_run(self, run_context):
        if self._fetch_tensors is not None:
            return tf.train.SessionRunArgs(self.tensors)

    def after_run(self, run_context, run_values):
        if self._step_callback is not None:
            self._step += 1
            self._step_callback(self._model, dict(zip(self.names, run_values.results)), self._step)

class Model(object):
    current = None
    
    def __init__(self, model_fn, model_dir, 
        config=None, run_config=None, warm_start_from=None, params={}, delete_existing=False):

        os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

        self.__config = config        
        self.__epoch = 0
        self.__init = None
        self.__feed_dict = None
        self.__last_feed_dict = None
        self.__results = None
        self.__epoch_bar = None
        self.__step_bar = None
        self.__data_parser = None
        
        self.__classifier = None
        self.__callbacks = []

        self.__clean = []

        # Set up a RunConfig to only save checkpoints once per training cycle.
        if run_config is None:
            run_config = tf.estimator.RunConfig() \
                .replace(save_checkpoints_secs=1e9)

        run_config = run_config \
            .replace(session_config=self.__config)

        if delete_existing:
            try: shutil.rmtree(model_dir)
            except: pass

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
        for fnc in self.__clean:
            fnc()
        Model.current = None       

    def data(self, data_parser):
        self.__data_parser = data_parser
        return self

    def clean(self, what):
        assert callable(what), "Argument should be callable"
        self.__clean.append(what)

    def _estimator_hook(self, func, steps, callback, log=None, summary=None, hooks=None):
        def hook(model, step, hooks):
            results = func(epochs=1, log=log, summary=summary, hooks=hooks, leave_bar=False)
            callback(results)

        self.__callbacks.append(tf.train.CheckpointSaverHook(
            checkpoint_dir=self.classifier().model_dir,
            save_steps=steps
        ))

        self.add_callback(steps, lambda model, step: hook(model, step, hooks))

    def eval_hook(self, steps, eval_callback, log=None, summary=None):
        eval_callback.set_model(self)        
        self._estimator_hook(
            self.evaluate, 
            steps, 
            eval_callback.aggregate_callback, 
            log, 
            summary,
            [eval_callback]
        )

    def predict_hook(self, steps, callback, log=None, summary=None):
        self._estimator_hook(
            self.predict, 
            steps, 
            callback, 
            log, 
            summary
        )
    
    def train(self, epochs, epochs_per_eval=None, eval_callback=None, eval_log=None, eval_summary=None):
        self.__epoch_bar = bar(total=epochs)
        self.__step_bar = bar()
        self.__callbacks += [TqdmHook(self.__step_bar)]

        if args.debug:
            try:
                get_ipython()
                raise Exception("Debugging must be done from command line")
            except:
                pass

            self.__callbacks += [tf_debug.LocalCLIDebugHook()]

        for self.__epoch in range(0, epochs):
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

            self.__epoch_bar.update(1)

            # Try to do an eval
            if isinstance(epochs_per_eval, int) and eval_callback is not None:
                if self.__epoch % epochs_per_eval == 0:
                    if self.__data_parser.has(tf.estimator.ModeKeys.EVAL):
                        results = self.evaluate(epochs=1, log=eval_log, summary=eval_summary, leave_bar=False)
                        eval_callback(results)
                    else:
                        print('You have no `evaluation` dataset')

    def predict(self, epochs, log=None, summary=None, hooks=None, leave_bar=True):
        self.__step_bar = bar(leave=leave_bar)
        hooks = hooks or []
        hooks += [TqdmHook(self.__step_bar)]

        if log is not None:
            hooks.append(tf.train.LoggingTensorHook(
                tensors=log,
                every_n_iter=1
            ))

        if summary is not None:
            hooks.append(CustomSummarySaverHook(
                summary_op=summary,
                save_steps=1,
                output_dir=os.path.join(self.classifier().model_dir, 'pred')
            ))

        return self.classifier().predict(
            input_fn=lambda: self.__data_parser.input_fn(mode=tf.estimator.ModeKeys.PREDICT, num_epochs=epochs),
            hooks=hooks
        )

    def evaluate(self, epochs, eval_callback=None, log=None, summary=None, hooks=None, leave_bar=True):
        self.__step_bar = bar(leave=leave_bar)
        hooks = hooks or []
        hooks += [TqdmHook(self.__step_bar)]

        if eval_callback is not None:
            eval_callback.set_model(self)
            hooks += [eval_callback]

        if log is not None:
            hooks.append(tf.train.LoggingTensorHook(
                tensors=log,
                every_n_iter=1
            ))

        if summary is not None:
            hooks.append(CustomSummarySaverHook(
                summary_op=summary,
                save_steps=1,
                output_dir=os.path.join(self.classifier().model_dir, 'eval')
            ))

        results = self.classifier().evaluate(
            input_fn=lambda: self.__data_parser.input_fn(mode=tf.estimator.ModeKeys.EVAL, num_epochs=epochs),
            hooks=hooks
        )

        if eval_callback is not None:
            eval_callback.aggregate_callback(results)

        return results

    def generator_from_eval(self, interpreter, tensors):
        generatorSetup = GeneratorFromEval(self, interpreter, tensors)
        return generatorSetup.generator


class GeneratorFromEval(object):
    def __init__(self, model, interpreter, tensors):
        self.__eval_callback = EvalCallback(
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

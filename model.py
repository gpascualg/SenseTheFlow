from .config import bar
from . import tfhooks
from .internals import Thread

import types
import tensorflow as tf
from tensorflow.python import debug as tf_debug
import numpy as np
from queue import Queue
from types import SimpleNamespace
import shutil
import argparse
import os
import copy

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


# convenience class to access non-existing members
class DefaultNamespace(SimpleNamespace):
    def __getattribute__(self, name):
        try:
            val = SimpleNamespace.__getattribute__(self, name)
            return lambda _: val
        except:
            return lambda d = None: d


class DataParser(object):
    def __init__(self):
        self.__input_fn = dict([
            (tf.estimator.ModeKeys.TRAIN, []),
            (tf.estimator.ModeKeys.PREDICT, []),
            (tf.estimator.ModeKeys.EVAL, [])
        ])

    def has(self, mode):
        return bool(self.__input_fn[mode])

    def num(self, mode):
        return len(self.__input_fn[mode])

    def from_generator(self, generator, output_types, output_shapes=None, parser_fn=None,
        pre_shuffle=False, post_shuffle=False, flatten=False, 
        skip=None, num_samples=None, batch_size=1,
        mode=tf.estimator.ModeKeys.TRAIN, **kwargs):

        generator = {
            'generator': generator, 
            'output_types': output_types,
            'output_shapes': output_shapes
        }

        input_fn = lambda num_epochs: self.generator_input_fn(
            generator,  parser_fn=parser_fn,
            pre_shuffle=pre_shuffle, post_shuffle=post_shuffle, flatten=flatten, 
            skip=skip, num_samples=num_samples, batch_size=batch_size,
            mode=mode, num_epochs=num_epochs
        )

        self.__input_fn[mode].append((input_fn, DefaultNamespace(**kwargs)))
        
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

    def generator_input_fn(self, generator, mode, parser_fn=None,
        pre_shuffle=False, post_shuffle=False, flatten=False, 
        skip=None, num_samples=None, batch_size=1, num_epochs=1):

        dataset = tf.data.Dataset.from_generator(**generator)

        # Pre-parsing shuffle
        if pre_shuffle:
            dataset = dataset.shuffle(buffer_size=pre_shuffle)

        if skip is not None:
            dataset = dataset.skip(skip)

        # No need to parse anything?
        if parser_fn is not None:
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
        assert bool(self.__input_fn[mode])
        for input_fn, args in self.__input_fn[mode]:
            yield (lambda: input_fn(num_epochs), args)

class EvalCallback(tf.train.SessionRunHook):
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
            return tf.train.SessionRunArgs(self.tensors)

    def after_run(self, run_context, run_values):
        if self._step_callback is not None:
            self._step += 1
            self._step_callback(self._model, dict(zip(self.names, run_values.results)), self._k, self._step)

class Model(object):
    instances = []
    
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
            delete_now = (delete_existing == 'force')

            if not delete_now:
                done = False
                while not done:
                    res = input('Do you really want to delete all models? [yes/no]: ').lower()
                    done = (res in ('y', 'yes', 'n', 'no'))
                    delete_now = (res in ('y', 'yes'))

            if delete_now:
                try: shutil.rmtree(model_dir)
                except: pass

        self.__classifier = tf.estimator.Estimator(
            model_fn=model_fn, model_dir=model_dir, config=run_config,
            warm_start_from=warm_start_from, params=params)
        
    def add_callback(self, steps, func):
        assert type(func) == types.FunctionType, "Expected a function in func"
        assert len(signature(func).parameters) == 2, "Expected func to have 2 parameters (model, step)"
        
        self.__callbacks.append(tfhooks.CallbackHook(steps, func, self))

    # Getters
    def epoch(self):
        return self.__epoch
    
    def bar(self):
        return self.__epoch_bar, self.__step_bar
    
    def classifier(self):
        return self.__classifier

    def __enter__(self):
        Model.instances.append(self)
        return self

    def __exit__(self, type, value, tb):
        self.clean()
        Model.instances.remove(self)

    def data(self, data_parser):
        self.__data_parser = data_parser
        return self

    def clean_fnc(self, what):
        assert callable(what), "Argument should be callable"
        self.__clean.append(what)

    def clean(self):
        for fnc in self.__clean:
            fnc()

    def redraw_bars(self):
        self.__epoch_bar = bar(total=epochs)
        self.__epoch_bar.update(self.__epoch)
        self.__step_bar = bar()

    def _estimator_hook(self, func, steps, callback=None, log=None, summary=None, hooks=None):
        def hook(model, step, hooks):
            results = func(epochs=1, log=log, summary=summary, hooks=hooks, leave_bar=False)
            if callback is not None:
                callback(results)

        self.__callbacks.append(tf.train.CheckpointSaverHook(
            checkpoint_dir=self.classifier().model_dir,
            save_steps=steps
        ))

        self.add_callback(steps, lambda model, step: hook(model, step, hooks))

    def eval_hook(self, steps, eval_callback, log=None, summary=None):
        eval_callback.set_model(self)        
        self._estimator_hook(
            lambda *args, **kwargs: self.evaluate(*args, eval_callback=eval_callback, **kwargs), 
            steps, 
            None, 
            log, 
            summary
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
        assert self.__data_parser.num(tf.estimator.ModeKeys.TRAIN) == 1, "One and only one train_fn must be setup through the DataParser"

        self.__epoch = 0
        self.redraw_bars()
        self.__callbacks += [tfhooks.TqdmHook(self)]

        if args.debug:
            self.__callbacks += [tf_debug.LocalCLIDebugHook()]

        train_epochs = epochs_per_eval or 1

        for self.__epoch in range(0, epochs, train_epochs):
            logger = tf.train.LoggingTensorHook(
                tensors={
                    'global_step/step': 'global_step'
                },
                every_n_iter=10
            )

            step_counter = tf.train.StepCounterHook(every_n_steps=10, output_dir=self.classifier().model_dir)

            for (input_fn, params) in self.__data_parser.input_fn(mode=tf.estimator.ModeKeys.TRAIN, num_epochs=train_epochs):
                self.classifier().train(
                    input_fn=input_fn,
                    hooks=self.__callbacks + [logger, step_counter]
                )

            self.__epoch_bar.update(train_epochs)

            # Try to do an eval
            if isinstance(epochs_per_eval, int):
                if self.__data_parser.has(tf.estimator.ModeKeys.EVAL):
                    evaluator = self.evaluate(epochs=1, eval_callback=eval_callback, log=eval_log, summary=eval_summary, leave_bar=False)
                    _ = list(evaluator)  # Force generator to iterate all elements
                else:
                    print('You have no `evaluation` dataset')

    def predict(self, epochs, log=None, summary=None, hooks=None, checkpoint_path=None, leave_bar=True):
        total = self.__data_parser.num(tf.estimator.ModeKeys.PREDICT)

        for k, (input_fn, params) in enumerate(self.__data_parser.input_fn(mode=tf.estimator.ModeKeys.PREDICT, num_epochs=epochs)):
            pred_hooks = params.hooks(hooks)
            if pred_hooks is not None:
                pred_hooks = copy.copy(pred_hooks)
            else:
                pred_hooks = []

            self.__step_bar = bar(leave=params.leave_bar(leave_bar))
            pred_hooks += [tfhooks.TqdmHook(self.__step_bar)]

            if log is not None:
                pred_hooks.append(tf.train.LoggingTensorHook(
                    tensors=params.log(log),
                    every_n_iter=1
                ))

            if summary is not None:
                name = 'pred' if total == 1 else 'pred-{}'.format(k)
                pred_hooks.append(args.CustomSummarySaverHook(
                    summary_op=params.summary(summary),
                    save_steps=1,
                    output_dir=os.path.join(self.classifier().model_dir, name)
                ))

        
            results = self.classifier().predict(
                input_fn=input_fn,
                hooks=pred_hooks,
                checkpoint_path=params.checkpoint_path(checkpoint_path)
            )

            # Keep yielding all results
            yield results

    def evaluate(self, epochs, eval_callback=None, log=None, summary=None, hooks=None, checkpoint_path=None, leave_bar=True):
        total = self.__data_parser.num(tf.estimator.ModeKeys.EVAL)

        for k, (input_fn, params) in enumerate(self.__data_parser.input_fn(mode=tf.estimator.ModeKeys.EVAL, num_epochs=epochs)):
            # Copy hooks to avoid modifying global hooks
            eval_hooks = params.hooks(hooks)
            if eval_hooks is not None:
                eval_hooks = copy.copy(eval_hooks)
            else:
                eval_hooks = []
            
            self.__step_bar = bar(leave=leave_bar)
            eval_hooks += [tfhooks.TqdmHook(self.__step_bar)]

            if log is not None:
                eval_hooks.append(tf.train.LoggingTensorHook(
                    tensors=params.log(log),
                    every_n_iter=1
                ))

            if summary is not None:
                name = 'eval' if total == 1 else 'eval-{}'.format(k)
                eval_hooks.append(tfhooks.CustomSummarySaverHook(
                    summary_op=params.summary(summary),
                    save_steps=1,
                    output_dir=os.path.join(self.classifier().model_dir, name)
                ))

            if eval_callback is not None:
                current_eval_callback = params.eval_callback(eval_callback)
                if isinstance(current_eval_callback, EvalCallback):
                    current_eval_callback.set_model(self)
                    current_eval_callback.set_k(k)
                    eval_hooks += [current_eval_callback]

            results = self.classifier().evaluate(
                input_fn=input_fn,
                hooks=eval_hooks,
                checkpoint_path=checkpoint_path
            )

            if eval_callback is not None:
                current_eval_callback = params.eval_callback(eval_callback)
                aggregate_callback = current_eval_callback

                if isinstance(current_eval_callback, EvalCallback):
                   aggregate_callback = eval_callback.aggregate_callback
                    
                aggregate_callback(self, k, results)

            # Keep yielding all results
            yield results

    def generator_from_eval(self, interpreter, tensors):
        generatorSetup = GeneratorFromEval(self, interpreter, tensors)
        return generatorSetup.generator


    # @https://blog.metaflow.fr/tensorflow-how-to-freeze-a-model-and-serve-it-with-a-python-api-d4f3596b3adc
    def freeze(self, output_node_names, frozen_model_name='frozen_model.pb'):
        model_dir = self.classifier().model_dir
        if not tf.gfile.Exists(model_dir):
            raise AssertionError(
                "Export directory doesn't exists. Please specify an export "
                "directory: %s" % model_dir)

        # We retrieve our checkpoint fullpath
        checkpoint = tf.train.get_checkpoint_state(model_dir)
        input_checkpoint = checkpoint.model_checkpoint_path
        
        # We precise the file fullname of our freezed graph
        absolute_model_dir = "/".join(input_checkpoint.split('/')[:-1])
        output_graph = absolute_model_dir + "/" + frozen_model_name

        # We clear devices to allow TensorFlow to control on which device it will load operations
        clear_devices = True

        # We start a session using a temporary fresh Graph
        with tf.Session(graph=tf.Graph()) as sess:
            # We import the meta graph in the current default Graph
            saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)

            # We restore the weights
            saver.restore(sess, input_checkpoint)

            if not output_node_names:
                print("You need to supply a list with one or more names of nodes to output_node_names. Either of")
                for op in tf.get_default_graph().get_operations(): 
                    print (op.name())
                return False

            # We use a built-in TF helper to export variables to constants
            output_graph_def = tf.graph_util.convert_variables_to_constants(
                sess, # The session is used to retrieve the weights
                tf.get_default_graph().as_graph_def(), # The graph_def is used to retrieve the nodes 
                output_node_names.split(",") # The output node names are used to select the usefull nodes
            ) 

            # Finally we serialize and dump the output graph to the filesystem
            with tf.gfile.GFile(output_graph, "wb") as f:
                f.write(output_graph_def.SerializeToString())
            print("%d ops in the final graph." % len(output_graph_def.node))

        return output_graph_def


class ExecutionWrapper(object):
    def __init__(self, model, thread):
        self.model = model
        self.thread = thread

        # Attach to model instances and clean laters
        Model.instances.append(self)
        self.model.clean_fnc(lambda: Model.instances.remove(self))

        # Start running now
        self.thread.start()

    # Expose some functions
    def terminate(self):
        if self.isAlive():
            self.thread.terminate()
            self.thread.join()
            self.model.clean()
            return True

        return False

    def isRunning(self):
        return self.thread.isAlive()

    def reattach(self):
        # Right now it is a simply wrapper to redraw, might do something else later on
        self.model.redraw_bars()

class AsyncModel(object):
    def __init__(self, *args, **kwargs):
        self.model = Model(*args, **kwargs)

    def wrap(self, fnc, *args, **kwargs):
        def inner_wrap(model, fnc, *args, **kwargs):
            fnc(*args, **kwargs)
            model.clean()

        t = Thread(target=inner_wrap, args=(self.model, fnc,) + args, kwargs=kwargs)
        return ExecutionWrapper(self.model, t)

    def __getattribute__(self, name):
        try:
            attr = object.__getattribute__(self, name)
            return attr
        except:
            pass
        
        attr = Model.__getattribute__(self.model, name)
        if name in ('train', 'test', 'evaluate'):
            return lambda *args, **kwargs: self.wrap(attr, *args, **kwargs)

        return attr

    def __enter__(self):
        # Do not add now, we might end up running nothing
        # Model.instances.append(self.model)
        return self

    def __exit__(self, type, value, tb):
        # Do not clean now, it could (potentially) blow up everything
        # self.model.clean()
        # Model.current = None
        pass


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

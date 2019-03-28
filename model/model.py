import os
import types
import shutil
import copy
import tqdm
import itertools
import tensorflow as tf
import time
import datetime
import json
import sys
from inspect import signature
from threading import Event
from tensorflow.python import debug as tf_debug

from . import tfhooks
from ..config import bar
from ..helper import DefaultNamespace, cmd_args
# from .utils import discover


def rmtree(path):
    # Shutil in Jupyter causes unexpected behaviour (tensorboard won't recognize new model)
    try:
        get_ipython().magic('sx rm -rf {} &>/dev/null'.format(path))
    except:
        try: shutil.rmtree(path)
        except: pass

def _wrap_results(results, wrappers, epochs):
    # If generator, yield all results first (to actually end the process)
    isGenerator = isinstance(results, types.GeneratorType)
    if isGenerator:
        return _wrap_generator(results, wrappers, epochs)
    
    # Results were already the returned value of the process, thus it has ended
    # Clean tqdm and return
    tqdm_wrapper = wrappers.pop(0)
    tqdm_wrapper.update_epoch(epochs)
    tqdm_wrapper.done()
    return results

def _wrap_generator(generator, wrappers, epochs):
    for element in generator:
        yield element

    # Clean after all yielding has been done
    tqdm_wrapper = wrappers.pop(0)
    tqdm_wrapper.update_epoch(epochs)
    tqdm_wrapper.done()



class Model(object):
    instances = []
    
    def __init__(self, model, model_dir, warm_start_from=None, params={},
        prepend_timestamp=False, append_timestamp=False, force_ascii_discover=False):

        self.__model = model


        # Post-loading
        self.__setup_fn = None
        self.__model_fn = model_fn
        self.__run_config = run_config
        self.__params = params
        self.__warm_start_from = warm_start_from

        self.__config = config
        self.__init = None
        self.__feed_dict = None
        self.__last_feed_dict = None
        self.__results = None
        self.__epoch_bar = None
        self.__step_bar = None
        self.__data_parser = None
        
        self.__classifier = None
        self.__callbacks = []
        self.__prerun_hooks = []
        self.__tqdm_hooks = {
            tf.estimator.ModeKeys.TRAIN: [],
            tf.estimator.ModeKeys.EVAL: [],
            tf.estimator.ModeKeys.PREDICT: []
        }

        self.stop_has_been_requested = False
        self.__is_loaded = Event()
        self.__clean = []
        self.__clean_once = []

        # Set up a RunConfig to only save checkpoints once per training cycle.
        if run_config is None:
            run_config = tf.estimator.RunConfig() \
                .replace(save_checkpoints_secs=1e9)

        run_config = run_config \
            .replace(session_config=self.__config)

        model_dir = os.path.normpath(model_dir)
        model_name = os.path.basename(model_dir)
        model_dir = model_dir[:-len(model_name)]

        # Discover models
        discover.discover(model_dir, model_name, self._continue_loading, 
                          prepend_timestamp, append_timestamp, force_ascii_discover)

    def _continue_loading(self, model, is_using_initialized_model):
        print('CONTINUE')
        print(self.__is_loaded.is_set())

        self.__is_using_initialized_model = is_using_initialized_model
        self.__model_data = model
        
        # Output some information
        print('Target model directory: {}'.format(self.__model_data.model_dir))
        self.__setup_fn(self)
        self.__is_loaded.set()

    def once_ready(self, setup_fn):
        if self.__is_loaded.is_set():
            setup_fn(self)
        else:
            self.__setup_fn = setup_fn        

    def _add_hook(self, hook):
        self.__callbacks.append(hook)
        
    def add_callback(self, steps, func):
        assert type(func) == types.FunctionType, "Expected a function in func"
        assert len(signature(func).parameters) == 2, "Expected func to have 2 parameters (model, step)"
        
        self._add_hook(tfhooks.CallbackHook(steps, func, self))

    # Getters
    def epoch(self, mode=tf.estimator.ModeKeys.TRAIN):
        assert self.__tqdm_hooks[mode], "No running classifier in mode {}".format(mode)
        # TODO(gpascualg): -1 might not be the one running now
        return self.__tqdm_hooks[mode][-1].epoch
    
    def bar(self, mode=tf.estimator.ModeKeys.TRAIN):
        return self.__tqdm_hooks[mode]
    
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

    def clean_once_fnc(self, what):
        assert callable(what), "Argument should be callable"
        self.__clean_once.append(what)

    def clean(self):
        for fnc in self.__clean:
            fnc()

        for fnc in self.__clean_once:
            fnc()

        self.__clean_once = []
        self.__is_loaded.clear()

    def redraw_bars(self, epochs=None, leave=True, force=False):
        for mode, wrappers in self.__tqdm_hooks.items():
            for wrapper in wrappers:
                wrapper.draw()

    def _estimator_hook(self, func, steps, callback=None, log=None, summary=None, hooks=None):
        def hook(model, step, hooks):
            results = func(epochs=1, log=log, summary=summary, hooks=hooks, leave_bar=False)
            if callback is not None:
                callback(results)

        self._add_hook(tf.compat.v1.train.CheckpointSaverHook(
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

    def add_prerun_hook(self, hook):
        self.__prerun_hooks.append(hook)

    def _execute_prerun_hooks(self, mode):
        for fnc in self.__prerun_hooks:
            fnc(self, mode)

    def _warn_not_initialized(self):
        if not self.__is_using_initialized_model:
            print('You are using a non-initialized model!', file=sys.stderr)
            print('Eval/predict results are expected to be random', file=sys.stderr)

    # Actual train implementation
    def _do_train_epoch(self, global_step, input_fn, params, optimizer):
        with bar() as steps_bar:
            for x, y in input_fn():
                global_step.assign(global_step + 1)
                with tf.GradientTape() as tape:
                    loss = self.__model_fn(x, y, global_step, tf.estimator.ModeKeys.TRAIN, params)

                watched_vars = tape.watched_variables()
                print(watched_vars)
                gradients = tape.gradient(loss, watched_vars)
                optimizer.apply_gradients(zip(gradients, watched_vars))

                steps_bar.set_description('Loss: {:.2f}'.format(float(loss)))
                steps_bar.update()

    def _wrap_n_epochs(self, epochs, epochs_per_step, callable, input_fn, params, *args):
        epochs_bar = bar(total=epochs)
        global_step = tf.Variable(0, dtype=tf.int32, trainable=False)
        x = tf.Variable(dtype=input_fn)

        for epoch in range(0, epochs, epochs_per_step):
            self._do_train_epoch(global_step, input_fn, params, *args)
            epochs_bar.update(epochs_per_step)
    
    def train(self, optimizer, epochs, epochs_per_eval=None, eval_callback=None, eval_log=None, eval_summary=None, production=False):
        assert self.__data_parser.num(tf.estimator.ModeKeys.TRAIN) >= 1, "At least one train_fn must be setup through the DataParser"
        self.__is_using_initialized_model = True
        self._execute_prerun_hooks(tf.estimator.ModeKeys.TRAIN)

        train_epochs = epochs_per_eval or epochs
        train_fn = self._wrap_n_epochs

        if production:
            train_fn = tf.function(train_fn)

        for (input_fn, params) in self.__data_parser.input_fn(mode=tf.estimator.ModeKeys.TRAIN, num_epochs=train_epochs):
            train_fn(epochs, train_epochs, self._do_train_epoch, input_fn, params, optimizer)

        # tqdm_wrapper = tfhooks.TqdmWrapper(epochs=epochs)
        # self.__tqdm_hooks[tf.estimator.ModeKeys.TRAIN].append(tqdm_wrapper)
        # tqdm_wrapper.draw()

        # if cmd_args.debug:
        #     self.__callbacks += [tf_debug.LocalCLIDebugHook()]

        # train_epochs = epochs_per_eval or epochs

        # for epoch in range(0, epochs, train_epochs):
        #     logger = tf.compat.v1.train.LoggingTensorHook(
        #         tensors={
        #             'global_step/step': 'global_step'
        #         },
        #         every_n_iter=10
        #     )

        #     step_counter = tf.compat.v1.train.StepCounterHook(every_n_steps=10, output_dir=self.classifier().model_dir)

        #     for (input_fn, params) in self.__data_parser.input_fn(mode=tf.estimator.ModeKeys.TRAIN, num_epochs=train_epochs):
        #         self.classifier().train(
        #             input_fn=input_fn,
        #             hooks=self.__callbacks + [tqdm_wrapper.create(), logger, step_counter]
        #         )

        #         # Stop request during training
        #         if self.stop_has_been_requested:
        #             return

        #     # Update epochs
        #     tqdm_wrapper.update_epoch(epoch + train_epochs)

        #     # Try to do an eval
        #     if isinstance(epochs_per_eval, int):
        #         if self.__data_parser.has(tf.estimator.ModeKeys.EVAL):
        #             evaluator = self.evaluate(epochs=1, eval_callback=eval_callback, log=eval_log, summary=eval_summary, leave_bar=False)
        #             _ = list(evaluator)  # Force generator to iterate all elements
        #         else:
        #             print('You have no `evaluation` dataset')

        #     # Stop request during eval
        #     if self.stop_has_been_requested:
        #         return

    def predict(self, epochs=1, log=None, summary=None, hooks=None, checkpoint_path=None, leave_bar=True):
        self._warn_not_initialized()
        self._execute_prerun_hooks(tf.estimator.ModeKeys.PREDICT)
        total = self.__data_parser.num(tf.estimator.ModeKeys.PREDICT)

        for k, (input_fn, params) in enumerate(self.__data_parser.input_fn(mode=tf.estimator.ModeKeys.PREDICT, num_epochs=epochs)):
            pred_hooks = params.hooks(hooks)
            if pred_hooks is not None:
                pred_hooks = copy.copy(pred_hooks)
            else:
                pred_hooks = []

            tqdm_wrapper = tfhooks.TqdmWrapper(epochs=params.epochs(epochs), leave=params.leave_bar(leave_bar))
            self.__tqdm_hooks[tf.estimator.ModeKeys.PREDICT].append(tqdm_wrapper)
            tqdm_wrapper.draw()
            pred_hooks += [tqdm_wrapper.create()]

            if params.log(log) is not None:
                pred_hooks.append(tf.compat.v1.train.LoggingTensorHook(
                    tensors=params.log(log),
                    every_n_iter=1
                ))

            if params.summary(summary) is not None:
                name = 'pred' if total == 1 else 'pred-{}'.format(k)
                pred_hooks.append(tfhooks.CustomSummarySaverHook(
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
            yield _wrap_results(results, self.__tqdm_hooks[tf.estimator.ModeKeys.PREDICT], params.epochs(epochs))
        

    def evaluate(self, epochs=1, eval_callback=None, log=None, summary=None, hooks=None, checkpoint_path=None, leave_bar=True):
        self._warn_not_initialized()
        self._execute_prerun_hooks(tf.estimator.ModeKeys.EVAL)
        total = self.__data_parser.num(tf.estimator.ModeKeys.EVAL)

        for k, (input_fn, params) in enumerate(self.__data_parser.input_fn(mode=tf.estimator.ModeKeys.EVAL, num_epochs=epochs)):
            # Copy hooks to avoid modifying global hooks
            eval_hooks = params.hooks(hooks)
            if eval_hooks is not None:
                eval_hooks = copy.copy(eval_hooks)
            else:
                eval_hooks = []

            tqdm_wrapper = tfhooks.TqdmWrapper(epochs=params.epochs(epochs), leave=params.leave_bar(leave_bar))
            self.__tqdm_hooks[tf.estimator.ModeKeys.EVAL].append(tqdm_wrapper)
            tqdm_wrapper.draw()
            eval_hooks += [tqdm_wrapper.create()]

            if params.log(log) is not None:
                eval_hooks.append(tf.compat.v1.train.LoggingTensorHook(
                    tensors=params.log(log),
                    every_n_iter=1
                ))

            if params.summary(summary) is not None:
                name = 'eval' if total == 1 else 'eval-{}'.format(k)
                eval_hooks.append(tfhooks.CustomSummarySaverHook(
                    summary_op=params.summary(summary),
                    save_steps=1,
                    output_dir=os.path.join(self.classifier().model_dir, name)
                ))

            if params.eval_callback(eval_callback) is not None:
                current_eval_callback = params.eval_callback(eval_callback)
                if isinstance(current_eval_callback, tfhooks.EvalCallbackHook):
                    current_eval_callback.set_model(self)
                    current_eval_callback.set_k(k)
                    eval_hooks += [current_eval_callback]

            results = self.classifier().evaluate(
                input_fn=input_fn,
                hooks=eval_hooks,
                checkpoint_path=params.checkpoint_path(checkpoint_path)
            )

            if params.eval_callback(eval_callback) is not None:
                current_eval_callback = params.eval_callback(eval_callback)
                aggregate_callback = current_eval_callback

                if isinstance(current_eval_callback, tfhooks.EvalCallbackHook):
                   aggregate_callback = eval_callback.aggregate_callback
                    
                aggregate_callback(self, k, results)

            # Keep yielding all results
            yield _wrap_results(results, self.__tqdm_hooks[tf.estimator.ModeKeys.EVAL], params.epochs(epochs))

    # @https://blog.metaflow.fr/tensorflow-how-to-freeze-a-model-and-serve-it-with-a-python-api-d4f3596b3adc
    def freeze(self, output_node_names, frozen_model_name='frozen_model.pb'):
        model_dir = self.classifier().model_dir
        if not tf.gfile.Exists(model_dir):
            raise AssertionError(
                "Export directory doesn't exists. Please specify an export "
                "directory: %s" % model_dir)

        # We retrieve our checkpoint fullpath
        checkpoint = tf.compat.v1.train.get_checkpoint_state(model_dir)
        input_checkpoint = checkpoint.model_checkpoint_path
        
        # We precise the file fullname of our freezed graph
        absolute_model_dir = "/".join(input_checkpoint.split('/')[:-1])
        output_graph = absolute_model_dir + "/" + frozen_model_name

        # We clear devices to allow TensorFlow to control on which device it will load operations
        clear_devices = True

        # We start a session using a temporary fresh Graph
        with tf.Session(graph=tf.Graph()) as sess:
            # We import the meta graph in the current default Graph
            saver = tf.compat.v1.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)

            # We restore the weights
            saver.restore(sess, input_checkpoint)

            if not output_node_names:
                print("You need to supply a list with one or more names of nodes to output_node_names. Either of")
                for op in tf.get_default_graph().get_operations(): 
                    try:
                        scope = op.name.split('/')[0]
                        if any(x in scope for x in ('gradients', 'report_uninitialized_variables', 'cond', 'metrics', 'WNAdam', 'save')):
                            continue
                    except:
                        pass
                    
                    print (op.name)
                return False

            # We use a built-in TF helper to export variables to constants
            output_graph_def = tf.graph_util.convert_variables_to_constants(
                sess, # The session is used to retrieve the weights
                tf.get_default_graph().as_graph_def(), # The graph_def is used to retrieve the nodes 
                output_node_names # The output node names are used to select the usefull nodes
            ) 

            # Finally we serialize and dump the output graph to the filesystem
            with tf.gfile.GFile(output_graph, "wb") as f:
                f.write(output_graph_def.SerializeToString())
            print("%d ops in the final graph." % len(output_graph_def.node))

        return output_graph_def

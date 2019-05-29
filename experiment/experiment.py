import tempfile
import os
import copy
import sys
import tensorflow as tf
import traceback as tb

from threading import Thread, Event, Lock, Condition
from concurrent.futures import ThreadPoolExecutor

from ..config import bar
from ..helper import DefaultNamespace, cmd_args
from .data import DataType, FetchMethod, UriType
from .utils import discover
from .mode import Mode


def default_config():
    gpu_options = tf.GPUOptions(allow_growth=True)
    return tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)

class Experiment(object):
    Instances = {}

    def __new__(cls, experiment_name, model_cls=None, on_data_ready=None, before_run=None, on_stop=None, persistent_path=None, params=None, max_concurrent_hooks=5):
        try:
            instance = Experiment.Instances[experiment_name]
        except:
            instance = object.__new__(cls)
            Experiment.Instances[experiment_name] = instance

        return instance
    
    def __init__(self, experiment_name, model_cls, on_data_ready=None, before_run=None, on_stop=None, persistent_path=None, params=None, max_concurrent_hooks=5):
        assert model_cls is not None, "It is required to pass a model class"
        assert not isinstance(model_cls, tf.keras.Model) and issubclass(model_cls, tf.keras.Model), "Model should be the class type, not an instance"

        # Params can be modified on any of the callbacks without affecting
        # the original object
        self.params = copy.copy(params or {})

        # Issue a warning if no parameters are set
        if not params:
            print("[WARNING] Defaulting to empty {} parameters", file=sys.stderr)

        # Private variables
        self.__model_cls = model_cls
        self.__data = []
        self.__gpu = None
        self.__gpu_lock = Lock()

        # Predefined path
        if persistent_path is None:
            self.__temp_path = tempfile.TemporaryDirectory()
            self.__persistent_path = self.__temp_path.name
        else:
            self.add_data(FetchMethod.COPY, UriType.PERSISTENT, persistent_path)

        # Discoverable
        self.__model_components = None
        self.__is_using_initialized_model = False

        # Callbacks
        self.__is_remote_execution = False
        self.__on_run = None
        self.__on_data_ready = on_data_ready
        self.__before_run = before_run
        self.__on_stop = on_stop

        # Async execution contexts
        self.__contexts = {mode: [] for mode in Mode}

        # Concurrent hook execution
        self.__pool = ThreadPoolExecutor(max_workers=max_concurrent_hooks)
        self.__pool_count = 0
        self.__max_concurrent_hooks = max_concurrent_hooks
        self.__pool_cond = Condition()

    def post_job(self, fn, *args, **kwargs):
        with self.__pool_cond:
            self.__pool_cond.wait_for(lambda: self.__pool_count < self.__max_concurrent_hooks)
            self.__pool_count += 1
            self.__pool.submit(self._monitor_job, fn, *args, **kwargs)

    def _monitor_job(self, fn, *args, **kwargs):
        fn(*args, **kwargs)
        with self.__pool_cond:
            self.__pool_count -= 1
            self.__pool_cond.notify()

    def _add_async_context(self, mode, context):
        if not isinstance(context, AsyncExecution):
            return

        if context in self.__contexts[mode]:
            return

        self.__contexts[mode].append(context)

    def get_context(self, mode):
        return self.__contexts[mode]

    def wait_ready(self):
        for mode in Mode:
            for context in self.__contexts[mode]:
                context.wait_ready()

    def wait_all(self):
        for mode in Mode:
            for context in self.__contexts[mode]:
                context.wait()

    def set_remote_execution(self):
        self.__is_remote_execution = True

    def assign_gpu(self, gpu):
        if not isinstance(gpu, (list, tuple)):
            gpu = (gpu,)

        self.__gpu = {g: 0 for g in gpu}

    def add_data(self, method, uri_type, uri):
        if uri_type == UriType.PERSISTENT:
            if any(x.UriType == uri_type for x in self.__data):
                raise AssertionError("Only one persistent storage per experiment is allowed")

            # We no longer need a temp path, use the provided one (on_data_ready sets it up)            
            self.__temp_path = None
            self.__persistent_path = None
        
        self.__data.append(DataType(method, uri_type, uri))

    def get_data(self):
        return self.__data

    def on_data_ready(self, data):
        # Save persistent directory
        if data.UriType == UriType.PERSISTENT:
            self.__persistent_path = data.LocalUri
        
        if self.__on_data_ready:
            self.__on_data_ready(self, data)

    def before_run(self):
        if self.__before_run:
            self.__before_run(self)

    def run(self):
        self.__on_run(self)

    def stop(self):
        if self.__on_stop:
            self.__on_stop(self)

    def get_gpu(self):
        assert self.__gpu is not None, "Got an invalid GPU"

        with self.__gpu_lock:
            gpu = min(self.__gpu.items(), key=lambda x: x[1])[0]
            self.__gpu[gpu] += 1
            return gpu

    def free_gpu(self, gpu):
        assert self.__gpu is not None, "Got an invalid GPU"

        with self.__gpu_lock:
            self.__gpu[gpu] -= 1

    def get_persistant_path(self):
        if not self.__is_remote_execution:
            for data in self.__data:
                if data.UriType == UriType.PERSISTENT:
                    return data.RemoteUri

        return self.__persistent_path

    def get_model_directory(self):
        if not self.__is_remote_execution:
            if self.__model_components is not None:
                return self.__model_components.model_dir
            
        return self.get_persistant_path()        

    def run_local(self, callback, prepend_timestamp=False, append_timestamp=False, force_ascii_discover=False, delete_existing=False, force_last=False):
        model_dir = os.path.normpath(self.get_persistant_path())
        model_name = os.path.basename(model_dir)
        model_dir = model_dir[:-len(model_name)]

        # Discover models
        return discover.discover(model_dir, model_name, lambda *args: self._continue_loading(callback, *args), 
                          prepend_timestamp, append_timestamp, delete_existing, force_ascii_discover, force_last)

    def run_remote(self, on_run):
        self.__on_run = on_run

    def _continue_loading(self, callback, model, is_using_initialized_model):
        self.__model_components = model
        self.__is_using_initialized_model = is_using_initialized_model
        callback(self)

    def assert_initialized(self):
        assert self.__is_using_initialized_model, "This model is not initialized"

    def __call__(self):
        assert self.__model_cls is not None, "Model is not configured"
        return self.__model_cls(self.params)

    def train(self, dataset_fn, epochs=1, config=None, warm_start_fn=None, hooks_fn=None, checkpoint_steps=1000, summary_steps=100, sync=False):
        run = ExperimentRun(self, Mode.TRAIN)
        context_or_none = run.run(dataset_fn, epochs=epochs, config=config, warm_start_fn=warm_start_fn, hooks_fn=hooks_fn, checkpoint_steps=checkpoint_steps, 
            summary_steps=summary_steps, sync=sync)
        self._add_async_context(Mode.TRAIN, context_or_none)
        return context_or_none

    def eval(self, dataset_fn, epochs=1, config=None, warm_start_fn=None, hooks_fn=None, summary_steps=100, sync=False):
        if not self.__is_using_initialized_model:
            print("[WARNING] Evaluating a non-trained model", file=sys.stderr)

        run = ExperimentRun(self, Mode.EVAL)
        context_or_none = run.run(dataset_fn, epochs=epochs, config=config, warm_start_fn=warm_start_fn, hooks_fn=hooks_fn, checkpoint_steps=None, 
            summary_steps=summary_steps, sync=sync)
        self._add_async_context(Mode.EVAL, context_or_none)
        return context_or_none

    def test(self, dataset_fn, epochs=1, config=None, warm_start_fn=None, hooks_fn=None, summary_steps=100, sync=False):
        if not self.__is_using_initialized_model:
            print("[WARNING] Testing a non-trained model", file=sys.stderr)

        run = ExperimentRun(self, Mode.TEST)
        context_or_none = run.run(dataset_fn, epochs=epochs, config=config, warm_start_fn=warm_start_fn, hooks_fn=hooks_fn, checkpoint_steps=None, 
            summary_steps=summary_steps, sync=sync)
        self._add_async_context(Mode.TEST, context_or_none)
        return context_or_none

class ExperimentOutput(object):
    def __init__(self, _sentinel=None, outputs=None, train_op=None, loss=None):
        assert _sentinel is None, "Please use named arguments, outputs=x, etc."
        self.outputs = outputs
        self.train_op = train_op
        self.loss = loss

class ExperimentHook(object):
    def __init__(self, name, steps, callback, concurrent=True, args=()):
        self.name = name
        self.__steps = steps
        self.__tensors = []
        self.__callback = callback
        self.__concurrent = concurrent
        self.__args = args
        self.__now = Event()
        self.__ready = Event()

    def ready(self, step):
        return self.__now.is_set() or (step % self.__steps) == 0

    def __call_callback(self, step, *args):
        try:
            self.__callback(step, *args, *self.__args)
        except Exception as e:
            # We can't have exceptions interrumpting the whole process
            if self.__concurrent:
                tb.print_exc()
        finally:
            self.__ready.set()

    def __call__(self, experiment, step, *args):
        self.__now.clear()
        self.__ready.clear()

        if self.__concurrent:
            experiment.post_job(self.__call_callback, step, *args)
        else:
            self.__call_callback(step, *args)

    def tensors(self):
        return self.__tensors

    def needs(self, tensor):
        self.__tensors.append(tensor)

    def trigger(self):
        self.__ready.clear()
        self.__now.set()

    def wait(self):
        self.__ready.wait()

class AsyncExecution(object):
    def __init__(self, experiment_run, *args, **kwargs):
        self.__experiment_run = experiment_run
        self.experiment = experiment_run.experiment
        self.__thread = Thread(target=experiment_run._run, args=args, kwargs=kwargs)
        self.__thread.start()

    def wait(self):
        try:
            self.__thread.join()
        except KeyboardInterrupt:
            return

    def wait_ready(self):
        try:
            self.__experiment_run.wait_ready()
        except KeyboardInterrupt:
            return
    
    def stop(self, block=True):
        self.__experiment_run.stop()
        if block:
            self.wait()

    def save(self, block=True):
        self.__experiment_run.save(block=block)

    def reattach(self):
        self.__experiment_run.reattach()

    def is_running(self):
        return self.__thread.is_alive()

class ExperimentRun(object):
    def __init__(self, experiment, mode):
        self.experiment = experiment
        self.mode = mode

        # To allow execution re-attaching
        self.__steps_bar = None
        self.__step = -1
        self.__stop = False

        # Trigger hook right now
        self.__checkpoint_hook = None
        self.__summaries_hook = None

        # Avoid weird issues with hook signaling
        self.__ready = Event()
        self.__run_lock = Lock()

    def reattach(self):
        with self.__run_lock:
            if self.__steps_bar is not None:
                self.__steps_bar.close()
        
            self.__steps_bar = bar()
            self.__steps_bar.update(self.__step)

    def save(self, block=True):
        assert self.__checkpoint_hook is not None, "First run the experiment"
        with self.__run_lock:
            self.__checkpoint_hook.trigger()

        if block:
            self.__checkpoint_hook.wait()

    def stop(self):
        with self.__run_lock:
            self.__stop = True

    # The user won't see this at all
    def run(self, *args, **kwargs):
        if kwargs['sync']:
            return self._run(*args, **kwargs)
        else:
            context = AsyncExecution(self, *args, **kwargs)
            return context

    def wait_ready(self):
        self.__ready.wait()

    @discover.GO.capture
    def _run(self, dataset_fn, epochs=1, config=None, warm_start_fn=None, hooks_fn=None, checkpoint_steps=1000, summary_steps=100, sync=None):
        # Get a GPU for execution
        gpu = self.experiment.get_gpu()

        with tf.Graph().as_default(), tf.device('/gpu:{}'.format(gpu)):
            with tf.Session(config=config or default_config()) as sess:
                # Create step and dataset iterator
                global_step = tf.train.get_or_create_global_step()
                dataset = dataset_fn(self.mode)
                itr = dataset.make_one_shot_iterator()

                # Test has no y
                if self.mode == Mode.TEST:
                    x = itr.get_next()
                    y = None
                else:
                    x, y = itr.get_next()

                # Input tensors to control deps
                input_tensors = []
                for v in (x, y):
                    if isinstance(v, (list, tuple)):
                        input_tensors += v

                    if isinstance(v, dict):
                        input_tensors += list(x.values())

                # Get outputs from model
                with tf.control_dependencies(input_tensors):
                    model = self.experiment()
                    outputs = model(x, y, self.mode, self.experiment.params)
                    assert isinstance(outputs, ExperimentOutput), "Output from model __call__ should be ExperimentOutput"
                    assert self.mode != Mode.TRAIN or outputs.train_op is not None, "During training outputs.train_op must be defined"

                # Any more hooks to be created
                hooks = (hooks_fn and hooks_fn(self.experiment, model, self.mode, x, y, outputs)) or []

                # Prep summaries and checkpoints
                model_dir = self.experiment.get_model_directory()
                merged = tf.summary.merge_all()
                writer = tf.summary.FileWriter(os.path.join(model_dir, self.mode.value), sess.graph)
        
                # Checkpoints
                saver = tf.train.Saver(filename=os.path.join(model_dir, 'model.ckpt'))
                
                # Run once
                sess.run(tf.global_variables_initializer())

                # Warm start hooks
                warm_start_fn and warm_start_fn(self.experiment, model, self.mode, sess)
                
                # Restore if there is anything to restore from
                ckpt = tf.train.get_checkpoint_state(model_dir) 
                if ckpt is not None:
                    print('Restoring from {}'.format(ckpt.model_checkpoint_path))
                    saver.restore(sess, ckpt.model_checkpoint_path)

                # Checkpoints?
                if checkpoint_steps is not None:
                    self.__checkpoint_hook = ExperimentHook(
                        name='Checkpoint',
                        steps=checkpoint_steps,
                        callback=lambda step: saver.save(sess, os.path.join(model_dir, 'model.ckpt'), global_step=step),
                        concurrent=False
                    )
                    hooks.append(self.__checkpoint_hook)
    
                # Summaries?
                if summary_steps is not None:
                    self.__summaries_hook = ExperimentHook(
                        name='Summaries',
                        steps=summary_steps,
                        callback=lambda step, merged: writer.add_summary(merged, step)
                    )
                    self.__summaries_hook.needs(merged)
                    hooks.append(self.__summaries_hook)

                # Standard requested tensors
                if self.mode == Mode.TRAIN:
                    standard_feed = [global_step, outputs.loss, outputs.train_op]
                else:
                    standard_feed = [global_step, outputs.loss, []]

                # Signal it is ready
                first = True
                self.__ready.set()

                # Up to epochs
                for epoch in bar(range(epochs)):
                    self.__steps_bar = bar()

                    try:
                        while not self.__stop:
                            # Lock
                            with self.__run_lock:
                                # Other requests of hooks
                                hooks_feed = [hook.tensors() for hook in hooks if hook.ready(self.__step + int(first or self.mode == Mode.TRAIN))]

                                # Run session
                                self.__step, loss, _, *hooks_output = sess.run(standard_feed + hooks_feed)
                            
                                # Update bar
                                self.__steps_bar.set_description('Loss: {:.2f}'.format(loss or '?'))
                                self.__steps_bar.update(1 if not first else self.__step)
                                first = False

                                # Call all hooks
                                for idx, hook in enumerate(hook for hook in hooks if hook.ready(self.__step)):
                                    hook(self.experiment, self.__step, *hooks_output[idx])

                    except tf.errors.OutOfRangeError:
                        # It's ok, one epoch done
                        pass
                    except KeyboardInterrupt:
                        # Exit gracefully
                        break
                    finally:
                        print('Closing bar')
                        self.__steps_bar.close()

                    if self.__stop:
                        break

        # Free current GPU
        self.experiment.free_gpu(gpu)


def keras_weight_loader(module, model, include_top, weights='imagenet'):
    if weights == 'imagenet':
        if include_top:
            weights_path = tf.keras.utils.get_file(
                module.WEIGHTS_PATH.split('/')[-1],
                module.WEIGHTS_PATH,
                cache_subdir='models')
        else:
            weights_path = tf.keras.utils.get_file(
                module.WEIGHTS_PATH_NO_TOP.split('/')[-1],
                module.WEIGHTS_PATH_NO_TOP,
                cache_subdir='models')
        
        model.load_weights(weights_path)

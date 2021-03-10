import tempfile
import os
import copy
import sys
import tensorflow as tf
import numpy as np
import traceback as tb
import semantic_version as sv
import itertools as it
import logging

# Internal tensorflow imports
from tensorflow.python.framework.func_graph import FuncGraph
from tensorflow.python.keras.backend import get_graph
from tensorflow.python.eager import context
from tensorflow.python.framework import errors_impl

from threading import Thread, Event, Lock, Condition
from concurrent.futures import ThreadPoolExecutor

from ..config import bar
from ..helper import DefaultNamespace, cmd_args
from .data import DataType, FetchMethod, UriType
from .utils import discover
from .mode import Mode, Hookpoint


STARS_SEP = '**********************************'


logger = logging.getLogger('SenseTheFlow')
ch = logging.StreamHandler()
ch.setLevel(level=logging.DEBUG)
formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

default_optimizer_config = tf.config.optimizer.get_experimental_options()

                    
def default_config(soft_device_placement=True, log_device_placement=False):
    tf.config.optimizer.set_experimental_options(default_optimizer_config)
    tf.debugging.set_log_device_placement(log_device_placement)
    tf.config.set_soft_device_placement(soft_device_placement)

    # Currently, memory growth needs to be the same across GPUs
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        logical_gpus = tf.config.list_logical_devices('GPU')
        logger.info('%d Physical GPUs, %d Logical GPUs', len(gpus), len(logical_gpus))
    else:
        raise NotImplementedError("No GPUs found")

def release_config(auto_mixed_precision=False):
    tf.config.optimizer.set_experimental_options({
        'layout_optimizer':                 True,
        'constant_folding':                 True,
        'shape_optimization':               True,
        'remapping':                        True,
        'arithmetic_optimization':          True,
        'dependency_optimization':          True,
        'loop_optimization':                True,
        'function_optimization':            True,
        'debug_stripper':                   True,
        'disable_model_pruning':            False,
        'scoped_allocator_optimization':    True,
        'pin_to_host_optimization':         True,
        'implementation_selector':          True,
        'auto_mixed_precision':             auto_mixed_precision
    })

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
            logger.critical("[WARNING] Defaulting to empty parameters", file=sys.stderr)

        # Private variables
        self.__model_cls = model_cls
        self.__data = []
        self.__devices = []
        self.__devices_usage = {}
        self.__devices_lock = Lock()

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
        self.__done_loading = Event()

        # Async execution contexts
        self.__contexts = {mode: [] for mode in Mode}

        # Concurrent hook execution
        self.__pool = ThreadPoolExecutor(max_workers=max_concurrent_hooks)
        self.__pool_count = 0
        self.__max_concurrent_hooks = max_concurrent_hooks
        self.__pool_cond = Condition()

        # Hooks
        self.__hooks = {
            Hookpoint.GRADIENT: [],
            Hookpoint.POST_INITIALIZATION: [],
            Hookpoint.LOOP: [],
            Hookpoint.EPOCH: []
        }

    def add_hook(self, hookpoint, hook, prepend=False, silent=False):
        # Check if it already exists
        for hk in self.__hooks[hookpoint]:
            if hk.name == hook.name and hk.mode == hook.mode:
                if not silent:
                    logger.critical(f'A hook with the name \'{hk.name}\' and mode \'{hk.mode}\' already exists in \'{hookpoint}\'')
                return
        
        if prepend:
            self.__hooks[hookpoint].insert(0, hook)
        else:
            self.__hooks[hookpoint].append(hook)

    def eval_every_n(self, steps, dataset_fn, input_shape, input_type):
        def _eval_fn(*args, **kwargs):
            self.eval(dataset_fn, None, input_shape=input_shape, input_type=input_type, leave_bars=False)
        
        hook = ExperimentHook('eval-every-n', steps, _eval_fn, mode=Mode.TRAIN)
        self.add_hook(Hookpoint.LOOP, hook)

    def eval_every_checkpoint(self, dataset_fn, input_shape, input_type, concurrent=True):
        def _eval_fn(*args, **kwargs):
            self.eval(dataset_fn, None, input_shape=input_shape, input_type=input_type, leave_bars=False)
        
        try:
            checkpoints_hook = next(x for x in self.get_hooks(Hookpoint.LOOP) if x.name == 'checkpoint')
            hook = ExperimentHook('eval-every-ckpt', checkpoints_hook.steps(), _eval_fn, concurrent=concurrent, mode=Mode.TRAIN)
            self.add_hook(Hookpoint.LOOP, hook, silent=True)
        except Exception:
            try:
                # Force the lookup, but we don't use it
                _ = next(x for x in self.get_hooks(Hookpoint.EPOCH) if x.name == 'checkpoint-epoch')
                hook = ExperimentHook.always('eval-every-ckpt', _eval_fn, concurrent=concurrent, mode=Mode.TRAIN)
                self.add_hook(Hookpoint.EPOCH, hook, silent=True)
            except Exception:
                raise NotImplementedError('To setup \'eval_every_checkpoint\' you need to either supply \'checkpoint_steps\' or set \'checkpoint_on_epoch\'')

    def get_hooks(self, hookpoint):
        return self.__hooks[hookpoint]

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
        self.__done_loading.wait()
        
        for mode in Mode:
            for context in self.__contexts[mode]:
                context.wait_ready()

    def wait_all(self):
        self.__done_loading.wait()
        
        for mode in Mode:
            for context in self.__contexts[mode]:
                context.wait()

    def show_progress(self):
        self.wait_ready()

        for mode in Mode:
            for context in self.__contexts[mode]:
                context.reattach()

    def set_remote_execution(self):
        self.__is_remote_execution = True

    def assign_gpu(self, gpus):
        if not isinstance(gpus, (list, tuple)):
            gpus = (gpus,)

        # Format gpus
        gpus = ['/gpu:{}'.format(x) for x in gpus]
        self.__devices.append(gpus)

        for device in gpus:
            self.__devices_usage[device] = 0
        
    def assign_device(self, devices):
        if not isinstance(devices, (list, tuple)):
            devices = (devices,)

        self.__devices.append(devices)

        for device in devices:
            self.__devices_usage[device] = 0

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

    def has_devices(self):
        with self.__devices_lock:
            return bool(self.__devices)

    def get_devices(self):
        with self.__devices_lock:
            assert self.__devices, "There is no devices available"

            devices = self.__devices.pop(0)
            for device in devices:
                self.__devices_usage[device] += 1
                
            return devices

    def free_devices(self, devices):
        if not isinstance(devices, (list, tuple)):
            devices = (devices,)
        
        with self.__devices_lock:
            for device in devices:
                assert device in self.__devices_usage, "Got an invalid device"
                self.__devices_usage[device] -= 1

            self.__devices.append(devices)

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

    def load_model_from_last_checkpoint(self, mode):
        model = self(mode)
        model_dir = self.get_model_directory()
        ckpt = tf.train.Checkpoint(net=model)
        manager = tf.train.CheckpointManager(ckpt, model_dir, max_to_keep=3)

        # Restore from checkpoint
        if manager.latest_checkpoint:
            restore_status = ckpt.restore(manager.latest_checkpoint)
            if not restore_status:
                raise NotImplementedError('There is no checkpoint yet')
    
            restore_status.assert_existing_objects_matched()
        else:
            raise NotImplementedError('There is no checkpoint yet')

        return model
      
    def run_local(self, prepend_timestamp=False, append_timestamp=False, force_ascii_discover=False, delete_existing=False, force_last=False):
        # Signal not ready
        self.__done_loading.clear()
        
        # Base directory and name
        model_dir = os.path.normpath(self.get_persistant_path())
        model_name = os.path.basename(model_dir)
        model_dir = model_dir[:-len(model_name)]

        # Discover models
        return discover.discover(self, model_dir, model_name, prepend_timestamp, append_timestamp, 
            delete_existing, force_ascii_discover, force_last)

    def run_remote(self, on_run):
        self.__on_run = on_run

    def on_discovered(self, output: discover.SelectionOutput):
        self.__model_components, self.__is_using_initialized_model = output.get()

        # Singnal ready
        self.__done_loading.set()

    def assert_initialized(self):
        assert self.__is_using_initialized_model, "This model is not initialized"

    def __call__(self, mode):
        assert self.__model_cls is not None, "Model is not configured"
        return self.__model_cls(mode, self.params)

    def train(self, dataset_fn, optimizer, epochs=1, pre_initialize_fn=None, post_initialize_fn=None, gradients_fn=None, report_steps_upper_bound=1000, 
             summary_steps=None, checkpoint_steps=1000, checkpoint_on_epoch=False, reset_metrics_at_epoch_start=True, call_on_epoch_before_run=True, input_shape=None, input_type=False, sync=False, use_bars=True, leave_bars=True):
        assert self.__done_loading.is_set(), "Not loaded yet"

        run = ExperimentRun(self, Mode.TRAIN, use_bars, leave_bars)
        context_or_none = run.run(dataset_fn, optimizer, epochs=epochs, pre_initialize_fn=pre_initialize_fn, post_initialize_fn=post_initialize_fn, gradients_fn=gradients_fn, 
            report_steps_upper_bound=report_steps_upper_bound, summary_steps=summary_steps, checkpoint_steps=checkpoint_steps, checkpoint_on_epoch=checkpoint_on_epoch, 
            reset_metrics_at_epoch_start=reset_metrics_at_epoch_start, call_on_epoch_before_run=call_on_epoch_before_run, input_shape=input_shape, input_type=input_type, sync=sync)
        self._add_async_context(Mode.TRAIN, context_or_none)
        return context_or_none

    def eval(self, dataset_fn, optimizer=None, epochs=1, pre_initialize_fn=None, post_initialize_fn=None, gradients_fn=None, report_steps_upper_bound=1000, 
             summary_steps=None, reset_metrics_at_epoch_start=True, call_on_epoch_before_run=False, input_shape=None, input_type=False, sync=False, use_bars=True, leave_bars=True):
        assert self.__done_loading.is_set(), "Not loaded yet"

        if not self.__is_using_initialized_model:
            logger.critical("[WARNING] Evaluating a non-trained model", file=sys.stderr)

        run = ExperimentRun(self, Mode.EVAL, use_bars, leave_bars)
        context_or_none = run.run(dataset_fn, optimizer, epochs=epochs, pre_initialize_fn=pre_initialize_fn, post_initialize_fn=post_initialize_fn, gradients_fn=gradients_fn, 
            report_steps_upper_bound=report_steps_upper_bound, summary_steps=summary_steps, checkpoint_steps=None, checkpoint_on_epoch=False, 
            reset_metrics_at_epoch_start=reset_metrics_at_epoch_start, call_on_epoch_before_run=call_on_epoch_before_run, input_shape=input_shape, input_type=input_type, sync=sync)
        self._add_async_context(Mode.EVAL, context_or_none)
        return context_or_none

    def test(self, dataset_fn, optimizer=None, epochs=1, pre_initialize_fn=None, post_initialize_fn=None, gradients_fn=None, report_steps_upper_bound=1000, 
             summary_steps=None, reset_metrics_at_epoch_start=True, call_on_epoch_before_run=False, input_shape=None, input_type=False, sync=False, use_bars=True, leave_bars=True):
        assert self.__done_loading.is_set(), "Not loaded yet"
        
        if not self.__is_using_initialized_model:
            logger.critical("[WARNING] Testing a non-trained model", file=sys.stderr)

        run = ExperimentRun(self, Mode.TEST, use_bars, leave_bars)
        context_or_none = run.run(dataset_fn, optimizer, epochs=epochs, pre_initialize_fn=pre_initialize_fn, post_initialize_fn=post_initialize_fn, gradients_fn=gradients_fn, 
            report_steps_upper_bound=report_steps_upper_bound, summary_steps=summary_steps, checkpoint_steps=None, checkpoint_on_epoch=False, 
            reset_metrics_at_epoch_start=reset_metrics_at_epoch_start, call_on_epoch_before_run=call_on_epoch_before_run, input_shape=input_shape, input_type=input_type, sync=sync)
        self._add_async_context(Mode.TEST, context_or_none)
        return context_or_none

    def _on_saved(self):
        self.__is_using_initialized_model = True

    def execute_in_summary_environment(self, mode, callback, *args, **kwargs):
        model_dir = self.get_model_directory()
        writer = tf.summary.create_file_writer(os.path.join(model_dir, mode.value))
        with writer.as_default():
            return callback(*args, **kwargs)

class ExperimentOutput(object):
    def __new__(cls, **kwargs):
        if sv.Version(tf.__version__).major == 1:
            return object.__new__(cls)

        return kwargs

    def __init__(self, _sentinel=None, outputs=None, train_op=None, loss=None):
        assert _sentinel is None, "Please use named arguments, outputs=x, etc."
        self.outputs = outputs
        self.train_op = train_op
        self.loss = tf.convert_to_tensor(loss)
        self.has_loss = loss is not None

class ExperimentHook(object):
    def __init__(self, name, steps, callback, concurrent=True, args=(), mode=Mode.ANY):
        self.name = name
        self.mode = mode
        self.__steps = steps
        self.__tensors = []
        self.__callback = callback
        self.__concurrent = concurrent
        self.__args = args
        self.__now = Event()
        self.__ready = Event()
        self.__skip_after_error = False

    @staticmethod
    def always(name, callback, concurrent=True, args=(), mode=Mode.ANY):
        return ExperimentHook(name, 1, callback, concurrent, args, mode)

    def steps(self):
        return self.__steps

    def ready(self, step, mode):
        if self.__skip_after_error:
            return False

        if self.mode != Mode.ANY and self.mode != mode:
            return False

        if not self.__steps:
            return False

        if self.__now.is_set():
            return True

        return (step % self.__steps) == 0

    def _call_callback(self, experiment, step, *args):
        try:
            return self.__callback(experiment, step, *args, *self.__args)
        
        # We can't have exceptions interrumpting the whole process
        except Exception as e:
            self.__skip_after_error = True
            logger.critical('Error on hook {}/{} -> Disabling hook'.format(self.name, self.mode), file=sys.stderr)
            tb.print_exc()
        finally:
            self.__ready.set()

        return None

    def __call__(self, experiment, step, *args):
        self.__now.clear()
        self.__ready.clear()

        if self.__concurrent:
            experiment.post_job(self._call_callback, experiment, step, *args)
        else:
            self._call_callback(experiment, step, *args)

    def set_args(self, *args):
        self.__args = args

    def tensors(self):
        return self.__tensors

    def needs(self, tensor):
        self.__tensors.append(tensor)

    def trigger(self):
        self.__ready.clear()
        self.__now.set()

    def wait(self):
        self.__ready.wait()

class SummaryHook(ExperimentHook):
    def __init__(self, summary_mode, name, steps, callback, concurrent=True, args=(), mode=Mode.ANY):
        super(SummaryHook, self).__init__(name=name, steps=steps, callback=callback, concurrent=concurrent, args=args, mode=mode)

        self.summary_mode = summary_mode

    def _call_callback(self, experiment, step, *args):
        model_dir = experiment.get_model_directory()
        writer = tf.summary.create_file_writer(os.path.join(model_dir, self.summary_mode.value))
        with writer.as_default():
            return super(SummaryHook, self)._call_callback(experiment, step, *args)

class AsyncExecution(object):
    def __init__(self, experiment_run, *args, **kwargs):
        self.__experiment_run = experiment_run
        self.experiment = experiment_run.experiment
        self.__model = None
        self.__thread = Thread(target=self._run, args=args, kwargs=kwargs)
        self.__thread.start()

    def _run(self, *args, **kwargs):
        self.__model = self.__experiment_run._run(*args, **kwargs)
        
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
      
    def model(self):
        if self.__model is None:
            raise NotImplementedError('Model is still running')
        return self.__model

    def load_model_from_last_checkpoint(self, mode):
        model = self.experiment(mode)
        model_dir = self.experiment.get_model_directory()
        ckpt = tf.train.Checkpoint(net=model)
        manager = tf.train.CheckpointManager(ckpt, model_dir, max_to_keep=3)

        # Restore from checkpoint
        if manager.latest_checkpoint:
            restore_status = ckpt.restore(manager.latest_checkpoint)
            if not restore_status:
                raise NotImplementedError('There is no checkpoint yet')
    
            restore_status.assert_existing_objects_matched()
        else:
            raise NotImplementedError('There is no checkpoint yet')

        return model

class ExperimentRun(object):
    def __init__(self, experiment, mode, use_bars, leave_bars):
        self.experiment = experiment
        self.mode = mode

        # Create bars now, won't work properly later
        self.use_bars = use_bars
        self.leave_bars = leave_bars
        if self.use_bars:
            self.__epochs_bar = bar(leave=leave_bars, ncols='100%')
            self.__steps_bar = bar(leave=leave_bars, ncols='100%')
        
        # To allow execution re-attaching
        self.__step = 0
        self.__stop = False

        # Trigger hook right now
        self.__checkpoint_hook = None
        self.__checkpoint_epoch_hook = None

        # Avoid weird issues with hook signaling
        self.__ready = Event()
        self.__run_lock = Lock()

    def reattach(self):
        with self.__run_lock:
            # Close current bar
            self.__steps_bar.close()
        
            # Create new bar
            self.__steps_bar = bar(leave=self.leave_bars, ncols='100%', initial=self.__step)

    def save(self, block=True):
        assert self.__checkpoint_hook is not None, "First run the experiment"
        with self.__run_lock:
            self.__checkpoint_hook.trigger()

        if block:
            self.__checkpoint_hook.wait()

    def __save(self, experiment, step, inputs, outputs, model, manager):
        save_path = manager.save()
        logger.info("Saved checkpoint for step {}: {}".format(step, save_path))
        experiment._on_saved()

    def stop(self):
        with self.__run_lock:
            self.__stop = True

    def __reset_steps_bar(self, description, amount=1):
        if self.use_bars:
            self.__steps_bar.close()
            self.__steps_bar = bar(leave=self.leave_bars, ncols='100%', initial=amount)
            self.__steps_bar.set_description(description)

    def __update_steps_bar(self, description, amount=1):
        if self.use_bars:
            self.__steps_bar.set_description(description)
            self.__steps_bar.update(amount)

    def __reset_epochs_bar(self, amount=1):
        if self.use_bars:
            self.__epochs_bar.close()
            self.__epochs_bar = bar(leave=self.leave_bars, ncols='100%', initial=amount)

    def __update_epochs_bar(self, amount=1):
        if self.use_bars:
            self.__epochs_bar.update(amount)

    def __close_bars(self):
        if self.use_bars:
            self.__steps_bar.close()
            self.__epochs_bar.close()

    # The user won't see this at all
    def run(self, *args, **kwargs):
        if kwargs['sync']:
            return self._run(*args, **kwargs)
        else:
            context = AsyncExecution(self, *args, **kwargs)
            return context

    def wait_ready(self):
        self.__ready.wait()

    def _run(self, *args, **kwargs):
        try:
            if sv.Version(tf.__version__).major == 1:
                raise NotImplementedError('Tensorflow 1.x support has been deprecated')
            else:
                return self._run_unsafe_2x(*args, **kwargs)
        finally:
            # Ensure ready is set if something fails
            # Might be already set, it is not a problem
            self.__ready.set()
            
        return None
      
    def reset_metrics(self, model):
      with tf.device('/cpu:0'):
        logger.info('Resetting metrics in mode {}'.format(self.mode.value))
        for metric in model.metrics:
            metric.reset_states()
            logger.debug('\tMetric {} has been reset'.format(metric.name))

    def _build(self, strategy, optimizer, model, input_shape, input_type):
        valid_types = (tuple, list, dict)
        if not isinstance(input_shape, valid_types):
            raise ValueError('Specified input shape is not one of the valid types. '
                            'Please specify a batch input shape of type tuple or '
                            'list of input shapes. User provided '
                            'input type: {}'.format(type(input_shape)))

        if context.executing_eagerly():
            graph = FuncGraph('build_graph')
        else:
            graph = get_graph()

        with graph.as_default():
            if (isinstance(input_shape, list) and
                all(d is None or isinstance(d, int) for d in input_shape)):
                input_shape = tuple(input_shape)
            
            if isinstance(input_shape, list):
                x = [tf.zeros(shape=shape, dtype=dtype) for shape, dtype in zip(input_shape, input_type)]
            elif isinstance(input_shape, dict):
                x = {
                    k: tf.zeros(shape=shape, dtype=input_type[k])
                    for k, shape in input_shape.items()
                }
            else:
                x = tf.zeros(shape=input_shape, dtype=input_type)

            def call_with_optimizer(x):
                model.call(x, training=self.mode == Mode.TRAIN, step=-1)

                # TODO(gpascualg): This is too much of a hack
                with tf.name_scope(optimizer._name):
                    # Create iteration if necessary.
                    with tf.init_scope():
                        optimizer._create_all_weights(model.trainable_variables)

            def call_no_optimizer(x):
                model.call(x, training=self.mode == Mode.TRAIN, step=-1)

            if optimizer is None:
                strategy.run(call_no_optimizer, args=(x,))
            else:
                strategy.run(call_with_optimizer, args=(x,))

    def _run_unsafe_2x(self, dataset_fn, optimizer, epochs=1, pre_initialize_fn=None, post_initialize_fn=None, gradients_fn=None, 
        report_steps_upper_bound=1000, summary_steps=None, checkpoint_steps=1000, checkpoint_on_epoch=False, reset_metrics_at_epoch_start=True, 
        call_on_epoch_before_run=True, input_shape=None, input_type=None, sync=None):
        # Failsafe
        model = None

        # Default gradients_fn
        if gradients_fn is None:
            gradients_fn = lambda gradients, variables, step: zip(gradients, variables)

        if not self.experiment.has_devices():
            self.__close_bars()
            logger.critical('No devices to execute in mode %s', self.mode.value)
            return None

        # Setup execution strategy
        devices = self.experiment.get_devices()

        # TODO(gpascualg): Is it worth doing this?
        strategy_cls = tf.distribute.MirroredStrategy
        if len(devices) == 1:
            devices = devices[0]
            strategy_cls = tf.distribute.OneDeviceStrategy

        strategy = strategy_cls(devices)

        with strategy.scope():
            # Create model
            model = self.experiment(self.mode)
            model_dir = self.experiment.get_model_directory()

            # Get dataset _input_fn
            dataset_input_fn = dataset_fn(self.mode, self.experiment.params)
            dataset = strategy.distribute_datasets_from_function(dataset_input_fn)
            iterator = iter(dataset)

            # Internal data
            stf = tf.Module()
            stf.step = tf.Variable(0, dtype=tf.int64, trainable=False)
            stf.epoch = tf.Variable(0, dtype=tf.int64, trainable=False)

            # Backward incompatible code
            assert getattr(model, "optimizer") is None, "Model must not have an `optimizer` member"

            # Create a writer, even if we don't end up using it
            writer = tf.summary.create_file_writer(os.path.join(model_dir, self.mode.value))

            # Create different kind of hooks for summaries and checkpoints
            #   before executing any initialize_fn or otherwise, create checkpoint hooks
            if summary_steps:
                # Summaries and signal ready
                def with_writer(experiment, step, inputs, outputs, model, manager):
                    with writer.as_default():
                        model.on_summary(step)
                        writer.flush()
                
                if hasattr(model, 'on_summary') and callable(model.on_summary):
                    summary_hook = ExperimentHook('summary', summary_steps, with_writer, concurrent=False, mode=self.mode)
                    self.experiment.add_hook(Hookpoint.LOOP, summary_hook, silent=True)
                else:
                    logger.critical('Summary is enabled but the model does not have an on_summary function')

            if checkpoint_steps:
                self.__checkpoint_hook = ExperimentHook('checkpoint', checkpoint_steps, self.__save, concurrent=False, mode=self.mode)
                self.experiment.add_hook(Hookpoint.LOOP, self.__checkpoint_hook, prepend=True, silent=True)

            if checkpoint_on_epoch:
                self.__checkpoint_epoch_hook = ExperimentHook.always('checkpoint-epoch', self.__save, concurrent=False, mode=self.mode)
                self.experiment.add_hook(Hookpoint.EPOCH, self.__checkpoint_epoch_hook, silent=True)

            # Define train/test functions
            def train_fn(data):
                with tf.GradientTape() as tape:
                    outputs = model(data, training=True, step=stf.step)
                    loss = outputs['loss']
                    if model.losses:
                        loss = loss + tf.add_n(model.losses)
                
                    loss = loss / strategy.num_replicas_in_sync
                    gradients = tape.gradient(loss, model.trainable_variables)
                    # TODO(gpascualg): Adding hooks here needs some work, it's not as trivial
                    optimizer.apply_gradients(gradients_fn(gradients, model.trainable_variables, stf.step))
                    return outputs

            def test_fn(data):
                return model(data, training=False, step=stf.step)

            # Checkpoint manager
            if self.mode == Mode.TRAIN:
                assert optimizer is not None, "Optimizer must be used in Mode.TRAIN"
                ckpt = tf.train.Checkpoint(stf=stf, optimizer=optimizer, net=model)
            else:
                optimizer = None
                ckpt = tf.train.Checkpoint(stf=stf, net=model)

            manager = tf.train.CheckpointManager(ckpt, model_dir, max_to_keep=3)

            # Do we have to warm start?
            postponed_assert = None
            restore_information = pre_initialize_fn and pre_initialize_fn(self.experiment, model, self.mode, ckpt)
            if restore_information:
                if isinstance(restore_information, (tuple, list)):
                    postponed_assert = restore_information[1]
                else:
                    postponed_assert = restore_information.assert_existing_objects_matched

            # Restore from checkpoint
            if manager.latest_checkpoint:
                restore_information = ckpt.restore(manager.latest_checkpoint)
                if self.mode == Mode.TRAIN:
                    postponed_assert = restore_information.assert_consumed
                else:
                    postponed_assert = restore_information.assert_existing_objects_matched

                self.__step = int(strategy.experimental_local_results(stf.step)[0])
                epoch = int(strategy.experimental_local_results(stf.epoch)[0])

                message = "Restored iter {} from {}".format(self.__step, manager.latest_checkpoint)
                logger.info(message)
                self.__reset_epochs_bar(epoch)
                self.__reset_steps_bar("Restored iter {} from {}".format(self.__step, manager.latest_checkpoint), self.__step)
            else:
                logger.info("Initializing from scratch: %s", model_dir)

            # If we have to load a model AND there is postponed information, reset metrics, eval on first, etc.
            needs_build = post_initialize_fn is not None or self.experiment.get_hooks(Hookpoint.POST_INITIALIZATION) or reset_metrics_at_epoch_start
            if postponed_assert is not None and needs_build:
                assert input_shape is not None and input_type is not None, "Using one of 'post_initialize_fn', 'Hookpoint.POST_INITIALIZATION' or 'reset_metrics_at_epoch_start' requires to submit an 'input_shape' and 'input_type'"

                # Build model
                self._build(strategy, optimizer, model, input_shape, input_type)

                # Assert loaded
                if postponed_assert is not None:
                    postponed_assert()
                    postponed_assert = None

            # Post initialize hooks (must have been initialized by now)
            post_initialize_fn and post_initialize_fn(self.experiment, model, self.mode, manager.latest_checkpoint)

            # Execute hooks, if any
            for hook in self.experiment.get_hooks(Hookpoint.POST_INITIALIZATION):
                if hook.ready(self.__step, self.mode):
                    hook(self.experiment, self.__step, None, None, model)

            # Hacky way to set the global scope function
            for x in it.chain((model,), model.submodules):
                setattr(x, '_global_scope', tf.name_scope(''))
            setattr(tf.keras.Model, 'global_scope', lambda self: self._global_scope)
            
            # Multi-steps function
            _step_fn = train_fn if self.mode == Mode.TRAIN else test_fn
            _number_of_steps = int(max(1, np.gcd.reduce(
                list(
                    it.chain([report_steps_upper_bound], (
                        x.steps() for x in self.experiment.get_hooks(Hookpoint.LOOP) if x.mode == Mode.ANY or x.mode == self.mode
                    ))
                )
            )))

            number_of_steps = tf.constant(_number_of_steps, dtype=tf.int32)
            increment_count = tf.constant(1 if self.mode == Mode.TRAIN else 0, dtype=tf.int64)

            logger.debug('Will run for {} steps at a time'.format(_number_of_steps))

            @tf.function
            def run_multiple_steps(iterator):
                # Do N-1 consecutive steps
                for _ in tf.range(number_of_steps - 1):
                    with tf.name_scope(''):
                        strategy.run(_step_fn, args=(next(iterator),))
                        stf.step.assign_add(increment_count)
                
                # One last step where we keep data and result
                data = next(iterator)
                result = strategy.run(_step_fn, args=(data,))
                stf.step.assign_add(increment_count)
                return data, result
            
            # Signal and go with writer context
            self.__ready.set()
            with writer.as_default():
                # Call now, before running
                if call_on_epoch_before_run and hasattr(model, 'on_epoch') and callable(model.on_epoch):
                    model.on_epoch(stf.step)
                    writer.flush()

                # Run it all
                for self.epoch in range(epochs):
                    if self.__stop:
                        break

                    # Reset iterator on second+ epochs
                    if self.epoch > 0:
                        iterator = iter(dataset)
                    
                    # Reset metrics
                    if reset_metrics_at_epoch_start:
                        self.reset_metrics(model)
                    
                    # Iterate all steps
                    while True:
                        # Do the actual iter
                        try:
                            data, outputs = run_multiple_steps(iterator)
                        except (tf.errors.OutOfRangeError, GeneratorExit, StopIteration):
                            # No more data
                            break
                        except (errors_impl.ResourceExhaustedError, tf.errors.ResourceExhaustedError):
                            logger.critical(STARS_SEP)
                            logger.critical(STARS_SEP)
                            logger.critical('Your network, in mode %s, has ran out of memory (OOM: tf.errors.ResourceExhaustedError)', self.mode.value)
                            logger.critical('Expect any other network running in device(s) %s to fail/crash/hang', str(devices))
                            logger.critical(STARS_SEP)
                            logger.critical(STARS_SEP)
                            break
                        except (errors_impl.InvalidArgumentError, tf.errors.InvalidArgumentError) as e:
                            if 'Tried to expand dim index 1 for tensor with 0 dimensions' in e.message:
                                logger.critical(STARS_SEP)
                                logger.critical('Your batch size in mode %s is not evenly divided by the number of replicas', self.mode.value)
                                logger.critical(STARS_SEP)
                            
                            tb.print_exc()
                            break
                        except Exception as e:
                            logger.critical('Fatal error during execution in mode %s', self.mode.value)
                            tb.print_exc()
                            break
                        
                        # Assert loaded in case we didn't do it before
                        if postponed_assert is not None:
                            postponed_assert()
                            postponed_assert = None
                            
                        # Increment step now
                        multisteps_count = int(strategy.experimental_local_results(stf.step)[0]) - self.__step
                        self.__step += multisteps_count

                        # Update tqdm
                        loss_other = 0.0 if not model.losses else tf.add_n(model.losses)
                        loss_model = strategy.reduce(tf.distribute.ReduceOp.SUM, outputs['loss'], axis=None)
                        loss = (loss_other + loss_model) / strategy.num_replicas_in_sync
                        self.__update_steps_bar('Loss/l2: {:.2f}/{:.2f}'.format(float(loss), float(loss_other)), multisteps_count)

                        # User hooks
                        for hook in self.experiment.get_hooks(Hookpoint.LOOP):
                            if hook.ready(self.__step, self.mode):
                                hook(self.experiment, self.__step, data, outputs, model, manager)

                        if self.__stop:
                            break

                    # Epoch done, do we have a callback?
                    if hasattr(model, 'on_epoch') and callable(model.on_epoch):
                        model.on_epoch(stf.step)
                        writer.flush()

                    # Update tqdm
                    self.__update_epochs_bar()
                    stf.epoch.assign_add(1)

                # Free current GPU
                self.__close_bars()
                self.experiment.free_devices(devices)

                # Execute hooks, if any
                for hook in self.experiment.get_hooks(Hookpoint.EPOCH):
                    if hook.ready(self.__step, self.mode):
                        hook(self.experiment, self.__step, None, None, model, manager)

        return model

import tempfile
import os
import copy
import sys
import tensorflow as tf
import traceback as tb
import semantic_version as sv

from threading import Thread, Event, Lock, Condition
from concurrent.futures import ThreadPoolExecutor

from ..config import bar
from ..helper import DefaultNamespace, cmd_args
from .data import DataType, FetchMethod, UriType
from .utils import discover, redirect
from .mode import Mode, Hookpoint

                    
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
            Hookpoint.LOOP: []
        }

    def add_hook(self, hookpoint, hook):
        self.__hooks[hookpoint].append(hook)

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
        # Signal not ready
        self.__done_loading.clear()
        
        # Base directory and name
        model_dir = os.path.normpath(self.get_persistant_path())
        model_name = os.path.basename(model_dir)
        model_dir = model_dir[:-len(model_name)]

        # Discover models
        return discover.discover(self, model_dir, model_name, lambda *args: self._continue_loading(callback, *args), 
                          prepend_timestamp, append_timestamp, delete_existing, force_ascii_discover, force_last)

    def run_remote(self, on_run):
        self.__on_run = on_run

    def _continue_loading(self, callback, model, is_using_initialized_model):
        # Save paths and call callback
        self.__model_components = model
        self.__is_using_initialized_model = is_using_initialized_model
        callback(self)
        
        # Singnal ready
        self.__done_loading.set()

    def assert_initialized(self):
        assert self.__is_using_initialized_model, "This model is not initialized"

    def __call__(self, mode):
        assert self.__model_cls is not None, "Model is not configured"
        return self.__model_cls(mode, self.params)

    def train(self, dataset_fn, epochs=1, config=None, pre_initialize_fn=None, post_initialize_fn=None, hooks_fn=None, checkpoint_steps=1000, summary_steps=100, sync=False):
        run = ExperimentRun(self, Mode.TRAIN)
        context_or_none = run.run(dataset_fn, epochs=epochs, config=config, pre_initialize_fn=pre_initialize_fn, post_initialize_fn=post_initialize_fn, 
            hooks_fn=hooks_fn, checkpoint_steps=checkpoint_steps, summary_steps=summary_steps, sync=sync)
        self._add_async_context(Mode.TRAIN, context_or_none)
        return context_or_none

    def eval(self, dataset_fn, epochs=1, config=None, pre_initialize_fn=None, post_initialize_fn=None, hooks_fn=None, summary_steps=100, sync=False):
        if not self.__is_using_initialized_model:
            print("[WARNING] Evaluating a non-trained model", file=sys.stderr)

        run = ExperimentRun(self, Mode.EVAL)
        context_or_none = run.run(dataset_fn, epochs=epochs, config=config, pre_initialize_fn=pre_initialize_fn, post_initialize_fn=post_initialize_fn, 
            hooks_fn=hooks_fn, checkpoint_steps=None, summary_steps=summary_steps, sync=sync)
        self._add_async_context(Mode.EVAL, context_or_none)
        return context_or_none

    def test(self, dataset_fn, epochs=1, config=None, pre_initialize_fn=None, post_initialize_fn=None, hooks_fn=None, summary_steps=100, sync=False):
        if not self.__is_using_initialized_model:
            print("[WARNING] Testing a non-trained model", file=sys.stderr)

        run = ExperimentRun(self, Mode.TEST)
        context_or_none = run.run(dataset_fn, epochs=epochs, config=config, pre_initialize_fn=pre_initialize_fn, post_initialize_fn=post_initialize_fn, 
            hooks_fn=hooks_fn, checkpoint_steps=None, summary_steps=summary_steps, sync=sync)
        self._add_async_context(Mode.TEST, context_or_none)
        return context_or_none

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
    def __init__(self, name, steps, callback, concurrent=True, args=()):
        self.name = name
        self.__steps = steps
        self.__tensors = []
        self.__callback = callback
        self.__concurrent = concurrent
        self.__args = args
        self.__now = Event()
        self.__ready = Event()
        self.__skip_after_error = False

    def ready(self, step):
        return not self.__skip_after_error and (self.__now.is_set() or (step % self.__steps) == 0)

    def _call_callback(self, experiment, step, *args):
        try:
            return self.__callback(step, *args, *self.__args)
        
        # We can't have exceptions interrumpting the whole process
        except Exception as e:
            self.__skip_after_error = True
            print('Error on hook {} -> Disabling hook'.format(self.name), file=sys.stderr)
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

        # Create bars now, won't work properly later
        self.__epochs_bar = bar()
        self.__steps_bar = bar()
        
        # To allow execution re-attaching
        self.__step = -1
        self.__stop = False

        # Trigger hook right now
        self.__checkpoint_hook = None
        self.__summaries_hook = None

        # Avoid weird issues with hook signaling
        self.__ready = Event()
        self.__run_lock = Lock()

        # Exit code
        self.__reason = None

    def reattach(self):
        with self.__run_lock:
            # Close current bar, if any
            if self.__steps_bar is not None:
                self.__steps_bar.close()
        
            # Create new bar
            self.__steps_bar = bar()
            self.__steps_bar.update(self.__step)
            
            # Recreate output capturing
            redirect.GlobalOutput(self.experiment).create()

    def save(self, block=True):
        assert self.__checkpoint_hook is not None, "First run the experiment"
        with self.__run_lock:
            self.__checkpoint_hook.trigger()

        if block:
            self.__checkpoint_hook.wait()

    def stop(self):
        with self.__run_lock:
            self.__stop = True

    def exit_reason(self):
        return self.__reason

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
                self._run_unsafe_1x(*args, **kwargs)
            else:
                self._run_unsafe_2x(*args, **kwargs)
        except:
            _, _, exc_traceback = sys.exc_info()
            self.__reason = tb.extract_tb(exc_traceback)
        finally:
            # Ensure ready is set if something fails
            # Might be already set, it is not a problem
            self.__ready.set()
        
    @redirect.capture_output
    def _run_unsafe_1x(self, dataset_fn, epochs=1, config=None, pre_initialize_fn=None, post_initialize_fn=None, hooks_fn=None, checkpoint_steps=1000, summary_steps=100, sync=None):
        # Get a GPU for execution
        gpu = self.experiment.get_gpu()

        with tf.Graph().as_default(), tf.device('/gpu:{}'.format(gpu)):
            with tf.Session(config=config or default_config()) as sess:
                # Create step and dataset iterator
                global_step = tf.train.get_or_create_global_step()
                dataset = dataset_fn(self.mode, self.experiment.params)
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
                    model = self.experiment(self.mode)
                    outputs = model((x, y), self.mode, self.experiment.params)
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
                pre_initialize_fn and pre_initialize_fn(self.experiment, model, self.mode, sess)
                sess.run(tf.global_variables_initializer())
                post_initialize_fn and post_initialize_fn(self.experiment, model, self.mode, sess)
                
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
                if summary_steps is not None and merged is not None:
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
                for self.__epoch in range(epochs):                    
                    try:
                        ended = False
                    
                        while not self.__stop:
                            # Lock
                            with self.__run_lock:
                                # Other requests of hooks
                                hooks_feed = [hook.tensors() for hook in hooks if hook.ready(self.__step + int(first or self.mode == Mode.TRAIN))]
                                
                                # Run session
                                self.__step, loss, _, *hooks_output = sess.run(standard_feed + hooks_feed)
                            
                                # Update bar
                                self.__steps_bar.set_description('Loss: {:.2f}'.format(loss))
                                self.__steps_bar.update(1 if not first else self.__step)
                                first = False

                                # Call all hooks
                                for idx, hook in enumerate(hook for hook in hooks if hook.ready(self.__step)):
                                    hook(self.experiment, self.__step, *hooks_output[idx])

                        ended = True
                        self.__epochs_bar.update(1)
                    except tf.errors.OutOfRangeError:
                        # It's ok, one epoch done
                        pass
                    except KeyboardInterrupt:
                        # Exit gracefully
                        break
                    finally:
                        # GUI Mode error?
                        if not ended and hasattr(self.__steps_bar, 'sp'):
                            self.__steps_bar.sp(bar_style='danger')
                        else:
                            self.__steps_bar.close()

                    if self.__stop:
                        break

        # Free current GPU
        self.experiment.free_gpu(gpu)

    @redirect.capture_output
    def _run_unsafe_2x(self, dataset_fn, epochs=1, config=None, pre_initialize_fn=None, post_initialize_fn=None, hooks_fn=None, checkpoint_steps=1000, summary_steps=100, sync=None):
        @tf.function
        def train_fn(data, step):
            with tf.GradientTape() as tape:
                outputs = model(data, step)
            
            gradients = tape.gradient(outputs['loss'], model.trainable_variables)
            # TODO(gpascualg): Adding hooks here needs some work, it's not as trivial
            model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            return outputs

        @tf.function
        def test_fn(data, step):
            return model(data, step)

        # Get a GPU for execution
        gpu = self.experiment.get_gpu()
        with tf.device('/gpu:{}'.format(gpu)):
            dataset = dataset_fn(self.mode, self.experiment.params)
            model = self.experiment(self.mode)
            step = tf.Variable(1, dtype=tf.int64)

            assert isinstance(getattr(model, "optimizer"), tf.keras.optimizers.Optimizer), "Model must have an `optimizer` member"

            pre_initialize_fn and pre_initialize_fn(self.experiment, model, self.mode, None)

            model_dir = self.experiment.get_model_directory()
            ckpt = tf.train.Checkpoint(step=step, optimizer=model.optimizer, net=model)
            manager = tf.train.CheckpointManager(ckpt, model_dir, max_to_keep=3)
            restore_status = ckpt.restore(manager.latest_checkpoint)

            if manager.latest_checkpoint:
                print("Restored from {}".format(manager.latest_checkpoint))
            else:
                print("Initializing from scratch.")

            writer = tf.summary.create_file_writer(os.path.join(model_dir, self.mode.value))
            self.__ready.set()

            def do_iter(data, step_fn):
                outputs = step_fn(data, step)
                hooks_fn and hooks_fn(self.experiment, model, self.mode, *data, outputs)
                step.assign_add(1)

                self.__steps_bar.set_description('Loss: {:.2f}'.format(float(outputs['loss'])))
                self.__steps_bar.update(1)

                if checkpoint_steps and (int(step) % checkpoint_steps) == 0:
                    save_path = manager.save()
                    print("Saved checkpoint for step {}: {}".format(int(step), save_path))

                for hook in self.experiment.get_hooks(Hookpoint.LOOP):
                    if hook.ready(int(step)):
                        hook(self.experiment, int(step), data, outputs, model)

            with writer.as_default():
                # Select function
                step_fn = train_fn if self.mode == Mode.TRAIN else test_fn

                # Manually call first iter, as to build the model
                for data in dataset:
                    do_iter(data, step_fn)
                    break

                # Make sure it is restored
                restore_status.assert_existing_objects_matched()

                # Post initialize hooks
                post_initialize_fn and post_initialize_fn(self.experiment, model, self.mode, None)

                # Run it all
                for self.epoch in range(epochs):
                    for data in dataset:
                        do_iter(data, step_fn)

        # Free current GPU
        self.experiment.free_gpu(gpu)

def keras_weight_path(module, include_top, weights='imagenet'):
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
        
        return weights_path

def keras_weight_loader(module, model, include_top, weights='imagenet', by_name=False):
    model.load_weights(keras_weight_path(module, include_top, weights), by_name=by_name)


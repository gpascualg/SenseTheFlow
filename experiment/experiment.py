import tempfile
import os
import tensorflow as tf

from threading import Thread

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

    def __new__(cls, experiment_name, model, on_data_ready=None, before_run=None, on_stop=None, persistent_path=None):
        try:
            instance = Experiment.Instances[experiment_name]
        except:
            instance = object.__new__(cls)
            Experiment.Instances[experiment_name] = instance

        return instance
    
    def __init__(self, experiment_name, model, on_data_ready=None, before_run=None, on_stop=None, persistent_path=None):
        print("INIT {}".format(model))
        # Vars
        self.__model = model
        self.__data = []
        self.__gpu = None

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

    def set_remote_execution(self):
        self.__is_remote_execution = True

    def assign_gpu(self, gpu):
        self.__gpu = gpu

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
            self.__on_data_ready(self, self.__model, data)

    def before_run(self):
        if self.__before_run:
            self.__before_run(self, self.__model)

    def run(self):
        self.__on_run(self, self.__model)

    def stop(self):
        if self.__on_stop:
            self.__on_stop(self, self.__model)

    def get_gpu(self):
        assert self.__gpu is not None, "Got an invalid GPU"

        return self.__gpu

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

    def run_local(self, callback, prepend_timestamp=False, append_timestamp=False, force_ascii_discover=False, delete_existing=False):
        model_dir = os.path.normpath(self.get_persistant_path())
        model_name = os.path.basename(model_dir)
        model_dir = model_dir[:-len(model_name)]

        # Discover models
        return discover.discover(model_dir, model_name, lambda *args: self._continue_loading(callback, *args), 
                          prepend_timestamp, append_timestamp, delete_existing, force_ascii_discover)

    def run_remote(self, on_run):
        self.__on_run = on_run

    def _continue_loading(self, callback, model, is_using_initialized_model):
        self.__model_components = model
        self.__is_using_initialized_model = is_using_initialized_model
        callback(self)

    def assert_initialized(self):
        assert self.__is_using_initialized_model, "This model is not initialized"

    def __call__(self, *args, **kwargs):
        assert self.__model is not None, "Model is not configured"
        return self.__model(*args, **kwargs)

    def train(self, dataset_fn, epochs=1, config=None, checkpoint_steps=1000, summary_steps=100, hooks=()):
        run = ExperimentRun(self, Mode.TRAIN)
        run.run(dataset_fn, epochs=epochs, config=config, checkpoint_steps=checkpoint_steps, 
                summary_steps=summary_steps, hooks=hooks)

    def eval(self, dataset_fn, epochs=1, config=None, summary_steps=100, hooks=()):
        run = ExperimentRun(self, Mode.EVAL)
        run.run(dataset_fn, epochs=epochs, config=config, checkpoint_steps=None, 
                summary_steps=summary_steps, hooks=hooks)

    def test(self, dataset_fn, epochs=1, config=None, summary_steps=100, hooks=()):
        run = ExperimentRun(self, Mode.PREDICT)
        run.run(dataset_fn, epochs=epochs, config=config, checkpoint_steps=None, 
                summary_steps=summary_steps, hooks=hooks)

class ExperimentOutput(object):
    def __init__(self, _sentinel=None, outputs=None, train_op=None, loss=None):
        assert _sentinel is None, "Please use named arguments, outputs=x, etc."
        self.outputs = outputs
        self.train_op = train_op
        self.loss = loss

    def _as_list(self):
        return (self.outputs, self.train_op, self.loss)

    def get_feed(self):
        return [x for x in self._as_list() if x is not None]

    def format_outputs(self, outputs):
        names = ['outputs', 'train_op', 'loss']
        names = [names[i] for i, x in enumerate(self._as_list()) if x is not None]

        return ExperimentOutput(**{
            name: value for name, value in zip(names, outputs)
        })

class ExperimentHook(object):
    def __init__(self, steps, callback, concurrent=True):
        self.__steps = steps
        self.tensors = []
        self.__callback = callback
        self.__concurrent = concurrent
        self.__now = False

    def ready(self, step):
        return self.__now or (step % self.__steps) == 0

    def __call__(self, step, *args):
        self.__now = False

        if self.__concurrent:
            thread = Thread(target=self.__callback, args=(step, *args))
            thread.start()
        else:
            self.__callback(step, *args)

    def needs(self, tensor):
        self.tensors.append(tensor)

    def trigger(self):
        self.__now = True

class ExperimentRun(object):
    def __init__(self, experiment, mode):
        self.experiment = experiment
        self.mode = mode

        # To allow execution re-attaching
        self.__steps_bar = None
        self.__step = -1

        # Trigger hook right now
        self.__checkpoint_hook = None
        self.__summaries_hook = None

    def reattach(self):
        if self.__steps_bar is not None:
            self.__steps_bar.close()
        
        self.__steps_bar = bar()
        self.__steps_bar.update(self.__step)

    def save(self):
        assert self.__checkpoint_hook is not None, "First run the experiment"
        self.__checkpoint_hook.trigger()

    def run(self, dataset_fn, epochs=1, config=None, checkpoint_steps=1000, summary_steps=100, hooks=()):
        with tf.Graph().as_default(), tf.device('/gpu:{}'.format(self.experiment.get_gpu())):
            with tf.Session(config=config or default_config()) as sess:
                # Create step and dataset iterator
                global_step = tf.train.get_or_create_global_step()
                dataset = dataset_fn()
                itr = dataset.make_one_shot_iterator()

                # Test has no y
                if self.mode == Mode.PREDICT:
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
                    outputs = self.experiment(x, y)
                    assert isinstance(outputs, ExperimentOutput), "Output from model __call__ should be ExperimentOutput"
        
                # Prep summaries and checkpoints
                model_dir = self.experiment.get_model_directory()
                merged = tf.summary.merge_all()
                writer = tf.summary.FileWriter(os.path.join(model_dir, self.mode.value), sess.graph)
        
                # Checkpoints
                saver = tf.train.Saver(filename=os.path.join(model_dir, 'model.ckpt'))
                
                # Run once
                sess.run(tf.global_variables_initializer())
                
                # Restore if there is anything to restore from
                ckpt = tf.train.get_checkpoint_state(os.path.join(model_dir, 'model.ckpt')) 
                if ckpt is not None:
                    saver.restore(sess, ckpt.model_checkpoint_path)

                # Checkpoints?
                hooks = list(hooks)
                if checkpoint_steps is not None:
                    self.__checkpoint_hook = ExperimentHook(
                        steps=checkpoint_steps,
                        callback=lambda step: saver.save(sess, os.path.join(model_dir, 'model.ckpt'), global_step=step),
                        concurrent=False
                    )
                    hooks.append(self.__checkpoint_hook)
    
                # Summaries?
                if summary_steps is not None:
                    self.__summaries_hook = ExperimentHook(
                        steps=summary_steps,
                        callback=lambda step, merged: writer.add_summary(merged, step)
                    )
                    self.__summaries_hook.needs(merged)
                    hooks.append(self.__summaries_hook)

                # First run to fix/update steps
                first = True

                # Up to epochs
                for epoch in bar(range(epochs)):
                    self.__steps_bar = bar()

                    try:
                        while True:
                            # Standard requested tensors
                            feed = [global_step, outputs.train_op, outputs.loss]

                            # Other requests of hooks
                            extra_inputs = 0
                            for hook in hooks:
                                if hook.ready(self.__step + 1):
                                    feed += [hook.tensors]
                                    extra_inputs += 1
                            
                            # Run session
                            self.__step, _, loss, *hooks_output = sess.run(feed)
                            
                            # Update bar
                            self.__steps_bar.set_description('Loss: {:.2f}'.format(loss or '?'))
                            self.__steps_bar.update(1 if not first else self.__step)
                            first = False

                            # Call all hooks
                            for idx, hook in enumerate(hooks):
                                if hook.ready(self.__step):
                                    hook(self.__step, *hooks_output[idx])

                    except tf.errors.OutOfRangeError:
                        # It's ok, one epoch done
                        pass
                    finally:
                        self.__steps_bar.close()
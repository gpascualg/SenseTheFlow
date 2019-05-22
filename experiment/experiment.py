import tempfile
import os

from threading import Thread

from ..config import bar
from ..helper import DefaultNamespace, cmd_args
from .data import DataType, FetchMethod, UriType
from .utils import discover
from .mode import Mode


def default_config(self):
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

    def train(self, dataset, epochs=1, config=None, checkpoint_steps=1000, summary_steps=100, hooks=()):
        run = ExperimentRun(self, Mode.TRAIN)
        run.run(dataset, epochs=epochs, config=config, checkpoint_steps=checkpoint_steps, 
                summary_steps=summary_steps, hooks=hooks)

    def eval(self, dataset, epochs=1, config=None, summary_steps=100, hooks=()):
        run = ExperimentRun(self, Mode.EVAL)
        run.run(dataset, epochs=epochs, config=config, checkpoint_steps=None, 
                summary_steps=summary_steps, hooks=hooks)

    def test(self, dataset, epochs=1, config=None, summary_steps=100, hooks=()):
        run = ExperimentRun(self, Mode.TEST)
        run.run(dataset, epochs=epochs, config=config, checkpoint_steps=None, 
                summary_steps=summary_steps, hooks=hooks)

class ExperimentOutput(object):
    def __init__(self, _sentinel=None, outputs=None, train_op=None, loss=None):
        assert _sentinel is None, "Please use named arguments, outputs=x, etc."
        self.outputs = outputs
        self.train_op = train_op
        self.loss = loss
        self.summaries = tf.summary.merge_all()

    def _as_list(self):
        return (self.outputs, self.train_op, self.loss, self.summaries)

    def get_feed(self):
        return [x for x in self._as_list() if x is not None]

    def format_outputs(self, outputs):
        names = ['outputs', 'train_op', 'loss', 'summaries']
        names = [names[i] for i, x in enumerate(self._as_list()) if x is not None]

        return ExperimentOutput(**{
            name: value for name, value in zip(names, outputs)
        })

class ExperimentHook(obect):
    def __init__(self, steps, callback, concurrent=True):
        self.steps = steps
        self.__callback = callback
        self.__concurrent = concurrent

    def __call__(self, step, output):
        if self.__concurrent:
            thread = Thread(target=save_plots, args=(step, output))
            thread.start()
        else:
            self.__callback(step, output)

class ExperimentRun(object):
    def __init__(self, experiment, mode):
        self.experiment = experiment
        self.mode = mode

    def run(self, dataset, epochs=1, config=None, checkpoint_steps=1000, summary_steps=100, hooks=()):
        with tf.Graph().as_default(), tf.device('/gpu:{}'.format(self.experiment.get_gpu())):
            with tf.Session(config=config or default_config()) as sess:
                # Create step and dataset iterator
                global_step = tf.train.get_or_create_global_step()
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
                    outputs = self.experiment(x, y)
                    assert isinstance(outputs, ExperimentOutput), "Output from model __call__ should be ExperimentOutput"
        
                # Prep summaries and checkpoints
                model_dir = self.experiment.get_model_directory()
                writer = tf.summary.FileWriter(os.path.join(model_dir, self.mode.value), sess.graph)
        
                # Checkpoints
                saver = tf.train.Saver(filename=os.path.join(model_dir, 'model.ckpt'))
                
                # Run once
                sess.run(tf.global_variables_initializer())
                
                # Restore if there is anything to restore from
                ckpt = tf.train.get_checkpoint_state(os.path.join(model_dir, 'model.ckpt')) 
                if ckpt is not None:
                    saver.restore(sess, ckpt.model_checkpoint_path)

                # First run to fix/update steps
                first = True
    
                # Up to epochs
                for epoch in bar(range(epochs)):
                    try:
                        with bar() as tqbar:
                            while True:
                                step, *run_outputs = sess.run([global_step, *outputs.get_feed()])
                                run_outputs = outputs.format_outputs(run_outputs)
                                
                                tqbar.set_description('Loss: {:.2f}'.format(run_outputs.loss or '?'))
                                tqbar.update(1 if not first else step)
                                first = False

                                if checkpoint_steps is not None and step % checkpoint_steps == 0:
                                    saver.save(
                                        sess,
                                        os.path.join(model_dir, 'model.ckpt'),
                                        global_step=global_step
                                    )

                                if summary_steps is not None and step % summary_steps == 0:
                                    writer.add_summary(run_outputs.summaries, step)

                                for hook in hooks:
                                    if step % hook.steps == 0:
                                        hook(step, run_outputs)

                except tf.errors.OutOfRangeError:
                    # It's ok, one epoch done
                    pass

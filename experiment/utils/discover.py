import datetime
import itertools
import json
import os
import sys
import time
import traceback as tb
from types import SimpleNamespace
from shutil import rmtree
from functools import wraps
from threading import Lock

from ...config import is_jupyter


def _ask(message, valid):
    resp = None
    while resp is None:
        resp = input(message).lower()
        resp = resp if (resp in valid) else None
    return resp

def _get_metadata( model_dir):
        data = []
        try:
            with open(os.path.join(model_dir, '.sensetheflow')) as fp:
                data = json.load(fp)
        except:
            pass

        return data
    
def _dump_metadata(model_dir, data):
    if not os.path.exists(model_dir):
        raise FileNotFoundError('Target directory <{}> does not exist'.format(model_dir))
    
    with open(os.path.join(model_dir, '.sensetheflow'), 'w') as fp:
        json.dump(data, fp)

def _create_model(model_dir, model_name, prepend_timestamp, append_timestamp):
    assert not (prepend_timestamp and append_timestamp), "Timestamp can either be appended, prepended or none of them, but not both"
    
    original_path = model_dir
    timestamp = time.time()

    if prepend_timestamp or append_timestamp:
        value = datetime.datetime.fromtimestamp(timestamp)
        value = value.strftime('%Y%m%d-%H%M%S')

        if append_timestamp:
            model_components = [model_name, value]
            model_dir = os.path.join(model_dir, model_name, value) 
        else:
            model_components = [value, model_name]
            model_dir = os.path.join(model_dir, value, model_name)
    else:
        model_components = [model_name]
        model_dir = os.path.join(model_dir, model_name)
    
    model = {
        'components': model_components,
        'timestamp': timestamp
    }

    data = _get_metadata(original_path)
    data += [model]
    _dump_metadata(original_path, data)

    return SimpleNamespace(
        model_dir=model_dir,
        **model
    )

def _is_path_valid_model(path):
    return os.path.exists(os.path.join(path, 'checkpoint'))

def _iterate_candidate_models(model_dir, model_name):
    for model in _get_metadata(model_dir):
        if model_name not in model['components']:
            continue
        
        path = os.path.join(model_dir, *model['components'])
        if not _is_path_valid_model(path):
            continue

        yield SimpleNamespace(
            model_dir=path,
            **model
        )

def _get_candidate_models(model_dir, model_name):
    model_as_is = []
    initial_model_path = os.path.join(model_dir, model_name)

    if _is_path_valid_model(initial_model_path):
        model_as_is = SimpleNamespace(
            model_dir=initial_model_path,
            timestamp=time.time(),
            components=[model_name]
        )
        model_as_is = [model_as_is]

    models = itertools.chain(
        model_as_is,
        (x for x in _iterate_candidate_models(model_dir, model_name) if not model_as_is or x.model_dir != model_as_is.model_dir)
    )

    # Might be empty
    try:
        return sorted(models, reverse=True, key=lambda x: x.timestamp)
    except:
        return []

class GlobalOutput(object):
    Instance = None

    def __new__(cls):
        if GlobalOutput.Instance is None:
            GlobalOutput.Instance = object.__new__(cls)
        return GlobalOutput.Instance

    def __init__(self):
        self.__out = None
        self.__ip = None
        self.__old_stdout = None
        self.__old_stderr = None
        self.__count = 0
        self.__lock = Lock()

    def create(self):
        if self.__out is not None:
            self.__out.close()

        if is_jupyter():
            import ipywidgets as widgets
            self.__out = widgets.Output()

    def widget(self):
        return self.__out

    def capture(self, fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            if self.__out is None or not is_jupyter():
                def inner():
                    return fn(*args, **kwargs)
            
            if self.__out is not None:
                if not is_jupyter():
                    with self:
                        return inner()
                else:
                    @self.__out.capture()
                    def inner():
                        return fn(*args, **kwargs)
            
            return inner()
        return wrapper

    class Redirect(object):
        def __init__(self, fn):
            self.fn = fn

        def write(self, string):
            self.fn(string)

        def flush(self):
            pass
    
    def unredirect(self):
        if self.__old_stdout:
            sys.stdout = self.__old_stdout
            sys.stderr = self.__old_stderr
            self.__old_stdout = None

    def redirect(self):
        if self.__old_stdout:
            return

        self.__old_stdout = sys.stdout
        self.__old_stderr = sys.stderr
        sys.stdout = GlobalOutput.Redirect(self.__out.append_stdout)
        sys.stderr = GlobalOutput.Redirect(self.__out.append_stderr)

    def __enter__(self):
        with self.__lock:
            if self.__count == 0:
                self.redirect()
            self.__count += 1

    def __exit__(self, exc_type, exc_val, exc_tb):
        with self.__lock:
            self.__count -= 1
            if self.__count == 0:
                # Print exception manually, otherwise we won't see it
                if exc_type is not None:
                    tb.print_exc()
                self.unredirect()

GO = GlobalOutput()

def _discover_jupyter(model_dir, model_name, prepend_timestamp, append_timestamp, delete_existing, candidates, on_discovered):
    import ipywidgets as widgets
    from IPython.display import display

    options = [('/'.join(x.components) + ('' if idx > 0 else ' *'), idx) for idx, x in enumerate(candidates)]

    select = widgets.Select(
        options=[
            ('-- Select a model --', None),
            ('-- Create new --', -1),
            ('-- Remove prev and create new --', -2)
        ] + options,
        index=0,
        disabled=False
    )

    # Create a new widget
    GO.create()

    @GO.capture
    def on_change(change):
        if change['type'] == 'change' and change['name'] == 'value':
            idx = change['new']
            if idx == -2 and candidates:
                if delete_existing == 'force' or _ask('Are you sure? [yes/no]: ', ('yes', 'y', 'no', 'n')) in ('y', 'yes'):
                    rmtree(candidates[0].model_dir)
                else:
                    return

            initialized = idx not in (-1, -2)
            select.close()

            if not initialized:
                model = _create_model(
                    model_dir, 
                    model_name, 
                    prepend_timestamp,
                    append_timestamp
                )
            else:
                model = candidates[idx]

            # get_ipython is no longer available due to being called from an "unsafe" environment
            @GO.capture
            def forward():
                return on_discovered(model, initialized)
            forward()

    # Display widgets
    display(select)
    display(GO.widget()) 

    # Listen for changes
    select.observe(on_change)

def discover(model_dir, model_name, on_discovered, prepend_timestamp, append_timestamp, delete_existing=False, force_ascii=False, force_last=False):
    candidates = _get_candidate_models(model_dir, model_name)

    if not candidates:
        model = _create_model(
            model_dir, 
            model_name, 
            prepend_timestamp,
            append_timestamp
        )
        return on_discovered(model, False)

    if force_last:
       return on_discovered(candidates[0], False)

    if is_jupyter() and not force_ascii:
        return _discover_jupyter(
            model_dir,
            model_name,
            prepend_timestamp,
            append_timestamp,
            delete_existing,
            candidates,
            on_discovered
        )

    num_candidates = len(candidates)
    is_using_initialized_model = False
    must_create_model = True
    ignore = False

    for current_candidate in candidates:
        print('Found candidate model at: {}'.format(current_candidate.model_dir))
        time.sleep(0.2)
        delete_now = (delete_existing == 'force')

        if delete_existing:
            if not delete_now:
                res = _ask('Do you want to remove this model (remove), use it now (use) or ignore and display next (ignore)? [remove/use/ignore]: ', ('remove', 'u', 'use', 'ignore', 'i'))
                delete_now = (res in ('remove'))
                ignore = res in ('ignore', 'i')

            if delete_now:
                must_create_model = True
                rmtree(current_candidate.model_dir)
                break
        
        else:
            res = _ask('Do you want to use this model (yes) or check next (no)? [yes/no]: ', ('y', 'yes', 'n', 'no'))
            ignore = res in ('no', 'n')


        # Use current candidate
        if not delete_now and not ignore:
            must_create_model = False
            is_using_initialized_model = True
            model = current_candidate
            break

    if ignore:
        must_create_model = True
        res = _ask('There are no more models, do you want to create a new one? [yes/no]: ', ('yes', 'y', 'no', 'n'))
        if res not in ('y', 'yes'):
            raise RuntimeError('Exiting')
                
    if must_create_model:
        model = _create_model(
            model_dir, 
            model_name, 
            prepend_timestamp,
            append_timestamp
        )
    
    return on_discovered(model, is_using_initialized_model)

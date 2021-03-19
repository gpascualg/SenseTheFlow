import datetime
import itertools
import json
import os
import sys
import time
import traceback as tb
import tensorflow as tf
from types import SimpleNamespace
from shutil import rmtree

from ...config import is_jupyter
from .redirect import capture_output, forward, GlobalOutput


def _ask(message, valid):
    resp = None
    while resp is None:
        resp = input(message).lower()
        resp = resp if (resp in valid) else None
    return resp

def _get_metadata(model_dir):
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


class AlreadyHasValue(Exception):
    pass

class ValueNotSelected(Exception):
    pass

class SelectionOutput(object):
    def __init__(self):
        self.__has_output = False
        self.__output = None
    
    def on_value(self, value):
        if self.__has_output:
            raise AlreadyHasValue()

        self.__has_output = True
        self.__output = value

    def get(self):
        if not self.__has_output:
            raise ValueNotSelected()

        return self.__output

def _discover_jupyter(output: SelectionOutput, model_dir, model_name, prepend_timestamp, append_timestamp, delete_existing, candidates):
    import ipywidgets as widgets
    from IPython.display import display

    options = [(' |- ' + '/'.join(x.components) + ('' if idx > 0 else ' *'), idx) for idx, x in enumerate(candidates)]

    select = widgets.Select(
        options=[
            ('-', None),
            (' |- Create new model', -1),
            (' |- Remove previous and create new', -2),
            ('-', None)
        ] + options,
        index=None,
        disabled=False
    )

    def on_change(change):
        if change['type'] == 'change' and change['name'] == 'value':
            idx = change['new']
            if idx is None:
                return
            
            if idx == -2 and candidates:
                if delete_existing == 'force':
                    rmtree(candidates[0].model_dir)
                else:
                    select.index = None
                    select.options = (
                        ('-', None),
                        (' |- Confirm deleting {}'.format('/'.join(candidates[0].components) ), -3),
                        (' |- Abort', -4)
                    )
                    return
            elif idx == -3:
                rmtree(candidates[0].model_dir)
                
            initialized = idx not in (-1, -2, -3)
            select.close()
            
            # Aborted deletion
            if idx == -4:
                return

            if not initialized:
                model = _create_model(
                    model_dir, 
                    model_name, 
                    prepend_timestamp,
                    append_timestamp
                )
            else:
                model = candidates[idx]

            output.on_value((model, initialized))

    # Display widgets
    display(select)

    # Listen for changes
    select.index = None
    select.observe(on_change)

def discover(experiment, model_dir, model_name, prepend_timestamp, append_timestamp, delete_existing=False, force_ascii=False, force_last=False, force_new=False):
    output = SelectionOutput()
    candidates = _get_candidate_models(model_dir, model_name)
    
    if not candidates:
        print("No models found, creating new")

        model = _create_model(
            model_dir, 
            model_name, 
            prepend_timestamp,
            append_timestamp
        )
        output.on_value((model, False))
        return output

    if force_new:
        raise RuntimeError('A model with this name already exists')

    if force_last:
        print("Forcing last model")
        output.on_value((candidates[0], True))
        return output

    if is_jupyter() and not force_ascii:
        _discover_jupyter(
            output,
            model_dir,
            model_name,
            prepend_timestamp,
            append_timestamp,
            delete_existing,
            candidates
        )
        return output

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
    
    output.on_value((model, is_using_initialized_model))
    return output

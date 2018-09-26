import subprocess
import os


def spawn(source, target):
    return subprocess.Popen(['lsyncd', '-rsync', source, target])

class LSync(object):
    def __init__(self, model, target_folder, sync_train=False, sync_eval=True, copy_at_end=True, remove_at_end=True):
        self._model = model
        self._copy_at_end = copy_at_end
        self._remove_at_end = remove_at_end

        # Setup callback
        model.clean_fnc(self.on_end)

        # Setup daemon
        source_dir = self._source_dir = model.classifier().model_dir
        model_name = os.path.basename(os.path.normpath(self._source_dir))
        target_dir = self._target_dir = os.path.join(target_folder, model_name)

        if sync_eval:
            source_dir = os.path.join(source_dir, 'eval')
            target_dir = os.path.join(target_dir, 'eval')
            
        if sync_eval or sync_train:
            self._process = spawn(source_dir, target_dir)
        else:
            self._process = None

    def on_end(self):
        if self._process is not None:
            self._process.kill()
            self._process = None

        if self._copy_at_end and self._remove_at_end:
            subprocess.call(['mv', self._source_dir, self._target_dir])
        elif self._copy_at_end:
            subprocess.call(['cp', self._source_dir])
        elif self._remove_at_end:
            subprocess.call(['rm', self._source_dir])
            
    def __enter__(self):
        return self._model.__enter__()

    def __exit__(self, type, value, tb):
        return self._model.__exit__(type, value, tb)

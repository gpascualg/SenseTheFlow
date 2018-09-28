from ..async.internals import Thread
import subprocess
import os


def spawn(source, target):
    return subprocess.Popen(['lsyncd', '-nodaemon', '-rsync', source, target])

class LSync(object):    
    def __init__(self, model, target_folder, sync_train=True, copy_at_start=True, copy_at_end=True, remove_at_end=True):
        self._model = model
        self._copy_at_end = copy_at_end
        self._remove_at_end = remove_at_end
        self._sync_train = sync_train

        # Setup callback
        model.clean_fnc(self.on_end)

        # Setup daemon parameters
        self._source_dir = model.classifier().model_dir
        model_name = os.path.basename(os.path.normpath(self._source_dir))
        self._target_dir = os.path.join(target_folder, model_name)

        subprocess.call(['mkdir', '-p', self._source_dir])
        subprocess.call(['mkdir', '-p', self._target_dir])

        if copy_at_start:
            self.__thread = Thread(target=self.copy_and_init)
            self.__thread.start()
            self._model.add_prerun_hook(self.wait_initial_copy_done)
        else:
            self.__thread = None
            self.on_init_done()
        
    def copy_and_init(self):
        subprocess.call(['cp', '-r', self._target_dir, self._source_dir])
        self.on_init_done()

    def wait_initial_copy_done(self, model, mode):
        if self.__thread is not None:
            self.__thread.join()

    def on_init_done(self):
        if self._sync_train:
            self._process = spawn(self._source_dir, self._target_dir)
        else:
            self._process = None

    def on_end(self):
        if self.__thread is not None:
            self.__thread.join()
        
        if self._process is not None:
            self._process.kill()
            self._process = None

        if self._copy_at_end and self._remove_at_end:
            subprocess.call(['mv', self._source_dir, self._target_dir])
        elif self._copy_at_end:
            subprocess.call(['cp', self._source_dir, self._target_dir])
        elif self._remove_at_end:
            subprocess.call(['rm', self._source_dir])


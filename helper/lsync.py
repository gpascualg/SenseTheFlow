from ..async.internals import Thread
import subprocess
import os
import textwrap
import tempfile


class LSync(object):    
    def __init__(self, model, target_folder, sync_train=True, copy_at_start=True, copy_at_end=True, remove_at_end=True):
        self._model = model
        self._copy_at_end = copy_at_end
        self._remove_at_end = remove_at_end
        self._sync_train = sync_train

        # Setup callback
        model.clean_fnc(self.on_end)

        # Setup daemon parameters
        source_dir = model.classifier().model_dir
        self._model_name = os.path.basename(os.path.normpath(source_dir))
        target_dir = os.path.join(target_folder, self._model_name)

        logpath = os.path.join(tempfile.gettempdir(), self._model_name + '.lsyncd')
        self._logfile = open(logpath, 'w')
        self._source_dir = os.path.normpath(source_dir)
        self._target_dir = os.path.normpath(target_dir)

        self.call(['mkdir', '-p', self._source_dir])
        self.call(['mkdir', '-p', self._target_dir])

        if copy_at_start:
            self.__thread = Thread(target=self.copy_and_init)
            self.__thread.start()
            self._model.add_prerun_hook(self.wait_initial_copy_done)
        else:
            self.__thread = None
            self.on_init_done()

    def log(self, msg):
        if self._logfile is not None:
            self._logfile.write('{}\n'.format(msg))
            self._logfile.flush()

    def call(self, args):
        self.log(args)
        subprocess.call(args)

    def spawn(self, args, stdout=None):
        self.log(args)
        return subprocess.Popen(args, stdout=stdout)
        
    def copy_and_init(self):
        self.call(['cp', '-rT', self._target_dir, self._source_dir])
        self.on_init_done()

    def wait_initial_copy_done(self, model, mode):
        if self.__thread is not None:
            self.__thread.join()

    def on_init_done(self):
        if self._sync_train:
            settings = """
                settings {{
                    nodaemon = true,
                    inotifyMode = "CloseWrite or Modify"
                }}

                sync {{
                    default.rsync,
                    source = "{0}"
                    target = "{1}"
                }}
            """

            settingspath = os.path.join(tempfile.gettempdir(), self._model_name + '.settings')
            with open(settingspath, 'w') as fp:
                fp.write(
                    textwrap.dedent(
                        settings.format(self._source_dir, self._target_dir)
                    )
                )
                
            self._process = self.spawn(['lsyncd', settingspath], self._logfile)
        else:
            self._process = None

    def on_end(self):
        if self.__thread is not None:
            self.__thread.join()
        
        if self._process is not None:
            self._process.kill()
            self._process.wait()
            self._process = None

        if self._copy_at_end and self._remove_at_end:
            self.call(['mv', '-T', self._source_dir, self._target_dir])
        elif self._copy_at_end:
            self.call(['cp', '-rT', self._source_dir, self._target_dir])
        elif self._remove_at_end:
            self.call(['rm', '-rf', self._source_dir])

        if self._logfile is not None:
            self._logfile.flush()
            self._logfile.close()
            self._logfile = None

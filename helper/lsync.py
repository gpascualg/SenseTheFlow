from ..async.internals import Thread
from ..config import bar
import subprocess
import os
import re
import textwrap
import tempfile


class LSync(object):    
    def __init__(self, model, target_folder, sync_train=True, copy_at_start=True,
        copy_at_end=True, remove_at_end=True, verbose=True, use_symlink=True):
        self._model = model
        self._copy_at_end = copy_at_end
        self._remove_at_end = remove_at_end
        self._sync_train = sync_train
        self._verbose = verbose
        self._use_symlink = use_symlink

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

        self.log("[LSYNCD] Initial setup")
        self.call(['mkdir', '-p', self._source_dir])
        self.call(['mkdir', '-p', self._target_dir])

        if copy_at_start:
            self.__thread = Thread(target=self.copy_and_init)
            self.__thread.start()
            self._model.add_prerun_hook(self.wait_initial_copy_done)
        else:
            self.__thread = None
            self.on_init_done()

    def log(self, msg, endline="\n"):
        if self._logfile is not None:
            self._logfile.write('{}{}'.format(msg, endline))
            self._logfile.flush()

        if self._verbose:
            print('{}{}'.format(msg, endline))

    def call(self, args):
        self.log(args)
        return subprocess.call(args, stdout=self._logfile)

    def spawn(self, args):
        self.log(args)
        return subprocess.Popen(args, stdout=self._logfile)

    def rsync(self, source, target, description):
        source = source + os.sep
        target = target + os.sep

        proc = subprocess.Popen(
            ['rsync', '-Wa', '--stats', '--dry-run', source, target],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE
        )

        remainder = proc.communicate()[0]
        mn = re.findall(r'Number of files: (\d+)', remainder.decode('utf-8'))
        total_files = int(mn[0])

        proc = subprocess.Popen(
            ['rsync', '-Wav', '--progress', source, target],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE
        )

        with bar(total=total_files, leave=False) as progress:
            progress.set_description(description)
            last = 0
            while True:
                output = proc.stdout.readline().decode('utf-8')
                if not output:
                    if proc.poll() is not None:
                        progress.update(total_files - last)
                        break
                    else:
                        continue
                
                self.log(output, "")
                if 'to-chk' in output:
                    m = re.findall(r'to-chk=(\d+)/(\d+)', output)
                    num = int(m[0][1]) - int(m[0][0])

                    if last != num:
                        progress.update(num - last)
                        last = num
                    
                    if int(m[0][0]) == 0:
                        break
        
    def copy_and_init(self):
        self.log("[LSYNCD] Initial copy")
        
        if self._use_symlink:
            if os.path.islink(self._source_dir):
                self.call(['rm', self._source_dir])
        
        if not os.path.islink(self._target_dir):
            self.rsync(self._target_dir, self._source_dir, 'Initial copy')
        
        self.on_init_done()

    def wait_initial_copy_done(self, model, mode):
        if self.__thread is not None:
            self.__thread.join()

    def on_init_done(self):
        self._process = None

        if self._use_symlink:
            self.call(['rm', '-rf', self._target_dir])
            self.call(['ln', '-s', self._source_dir, self._target_dir])

        elif self._sync_train:
            settings = """
                settings {{
                    nodaemon = true,
                    inotifyMode = "CloseWrite or Modify"
                }}

                sync {{
                    default.rsync,
                    source = "{0}",
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
                
            self.log("[LSYNCD] Starting")
            self._process = self.spawn(['lsyncd', settingspath])

    def on_end(self):
        if self.__thread is not None:
            self.__thread.join()
        
        if self._process is not None:
            self._process.kill()
            self._process.wait()
            self._process = None

        if self._use_symlink:
            self.call(['rm', self._target_dir])

        if self._copy_at_end:
            self.rsync(self._source_dir, self._target_dir, 'Copy at end')
        
        if self._remove_at_end:
            self.call(['rm', '-rf', self._source_dir])

        if self._use_symlink:
            self.call(['ln', '-s', self._target_dir, self._source_dir])

        if self._logfile is not None:
            self._logfile.flush()
            self._logfile.close()
            self._logfile = None

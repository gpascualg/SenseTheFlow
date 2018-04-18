import os
import argparse
import re
import subprocess
from getpass import getpass


def subprocess_cmd(command, cin=None):
    print(command)
    process = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
    if cin is not None:
        process.communicate(cin)
    process.wait()
    return process.returncode


def main():
    parser = argparse.ArgumentParser(description='Build options.')
    parser.add_argument('--tag', required=True)
    parser.add_argument('--python-version', default='3')
    parser.add_argument('--dir', default='.')
    parser.add_argument('--tensorflow', action='store_true')
    parser.add_argument('--tensorflow-version', default='r1.0')
    parser.add_argument('--bazel-version', default='0.11.0')
    parser.add_argument('--caffe', action='store_true')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--opencl', action='store_true')
    parser.add_argument('--ssh', action='store_true')
    parser.add_argument('--push', action='store_true')
    parser.add_argument('--half-precision', action='store_true')

    args = parser.parse_args()

    data = {
        'python_version27': int(args.python_version) == 2,
        'build_tensorflow': int(args.tensorflow),
        'tensorflow_version': args.tensorflow_version,
        'bazel_version': args.bazel_version,
        'build_caffe': int(args.caffe),
        'base': 'nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04' if args.gpu else 'ubuntu:16.04',
        'use_cuda': 1 if args.gpu else 0,
        'use_opencl': int(args.opencl),
        'compute_capabilities': '3.5,5.2,6.1' if args.half_precision else '3.5,5.2',
        'ssh': int(args.ssh)
    }

    write_stack = [(None, True)]

    with open(os.path.join(args.dir, 'Dockerfile.template'), 'r') as fin:
        with open(os.path.join(args.dir, 'Dockerfile'), 'w') as fout:
            for line in fin:
                match = re.match(r"\[\[if (\w+)\]\]", line)
                if match is not None:
                    if not write_stack[-1][1]:
                        write_stack.append([match.group(1), None])
                    else:
                        write_stack.append([match.group(1), data[match.group(1)]])
                    continue

                match = re.match(r"\[\[endif\]\]", line)
                if match is not None:
                    write_stack.pop(len(write_stack) - 1)
                    continue

                match = re.match(r"\[\[else\]\]", line)
                if match is not None:
                    if write_stack[-1][1] is not None:
                        write_stack[-1][1] = not write_stack[-1][1]
                    continue

                if write_stack[-1][1] is None or not write_stack[-1][1]:
                    continue

                match = re.match(r".*?({{(\w+)}}).*?", line)
                if match is not None:
                    line = line.replace(match.group(1), str(data[match.group(2)]))

                fout.write(line)

    if subprocess.call(['nvidia-docker', 'build', '-t', args.tag, args.dir]) == 0:
        print('')
        print("----- DONE -----")

        if args.push:
            #username = raw_input('Username: ')
            #password = getpass()

            if subprocess.call(['docker', 'login']) == 0:
                if subprocess.call(['docker', 'push', args.tag]) == 0:
                    print('')
                    print("----- PUSHED -----")
        


if __name__ == '__main__':
    main()

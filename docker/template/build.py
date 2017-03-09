import os
import argparse
import re
import subprocess
from getpass import getpass


def subprocess_cmd(command, cin=None):
    print command
    process = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
    if cin is not None:
        process.communicate(cin)
    process.wait()
    return process.returncode


def main():
    parser = argparse.ArgumentParser(description='Build options.')
    parser.add_argument('--tag', required=True)
    parser.add_argument('--tensorflow', action='store_true')
    parser.add_argument('--caffe', action='store_true')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--opencl', action='store_true')
    parser.add_argument('--push', action='store_true')

    args = parser.parse_args()
    
    data = {
        'build_tensorflow': int(args.tensorflow),
        'build_caffe': int(args.caffe),
        'base': 'nvidia/cuda:8.0-cudnn5-devel-ubuntu16.04' if args.gpu else 'ubuntu:16.04',
        'use_cuda': 1 if args.gpu else 0,
        'use_opencl': int(args.opencl)
    }

    write_stack = [(None, True)]

    with open('Dockerfile.template', 'r') as fin:
        with open('Dockerfile', 'w') as fout:
            for line in fin:
                match = re.match(r"\[\[if (\w+)\]\]", line)
                if match is not None:
                    write_stack.append([match.group(1), data[match.group(1)]])
                    continue

                match = re.match(r"\[\[endif\]\]", line)
                if match is not None:
                    write_stack.pop(len(write_stack) - 1)
                    continue

                match = re.match(r"\[\[else\]\]", line)
                if match is not None:
                    write_stack[-1][1] = not write_stack[-1][1]
                    continue

                if not write_stack[-1][1]:
                    continue

                match = re.match(r".*?({{(\w+)}}).*?", line)
                if match is not None:
                    line = line.replace(match.group(1), str(data[match.group(2)]))

                fout.write(line)

    if subprocess.call(['docker', 'build', '-t', args.tag, '.']) == 0:
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

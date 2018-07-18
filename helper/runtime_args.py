import argparse
from types import SimpleNamespace


def get_arguments():
    # If we are running inside Jupyter do nothing
    # Adding argparse inside a Jupyter notebook blows the whole
    # environment up (either case, we can not debug inside Jupyter)
    try:
        get_ipython()
        args = SimpleNamespace(debug=False)
    except:
        parser = argparse.ArgumentParser()
        parser.add_argument('--debug', default=False, action='store_true')
        args = parser.parse_args()

    return args

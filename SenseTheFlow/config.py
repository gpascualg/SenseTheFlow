import tqdm
from threading import Thread, Lock, Event

def is_jupyter():
    # If we are in Jupyter, load the notebook version
    # Otherwise default to text
    try:
        get_ipython
        return True
    except:
        return False


def get_bar():
    if is_jupyter():
        try:
            return tqdm.notebook.tqdm
        except:
            return tqdm.tqdm_notebook

    return tqdm.tqdm

bar = get_bar()

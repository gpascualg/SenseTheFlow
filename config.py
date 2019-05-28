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


bar = tqdm.tqdm_notebook if is_jupyter() else tqdm.tqdm

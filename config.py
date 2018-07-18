import tqdm

# If we are in Jupyter, load the notebook version
# Otherwise default to text
try:
    get_ipython
    bar = tqdm.tqdm_notebook
except:
    bar = tqdm.tqdm

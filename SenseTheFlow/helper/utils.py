
def recurse_layers(model):
    for x in model.layers:
        try:
            yield from recurse_layers(x)
        except:
            yield x

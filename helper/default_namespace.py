from types import SimpleNamespace


# convenience class to access non-existing members
class DefaultNamespace(SimpleNamespace):
    def __getattribute__(self, name):
        try:
            val = SimpleNamespace.__getattribute__(self, name)
            return lambda _: val
        except:
            return lambda d = None: d

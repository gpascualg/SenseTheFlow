
class RocksConcat(object):
    def __init__(self, stores, elements_per_iter):
        self.stores = stores
        self.elements_per_iter = elements_per_iter

    def iterate(self, cyclic=True):
        has_done_epoch = [False] * len(self.stores)
        itrs = [store.iterate(cyclic=cyclic) for store in self.stores]

        while True:
            if all(has_done_epoch) and not cyclic:
                raise StopIteration()

            for pos in range(len(self.stores)):
                results = []
                itr = itrs[pos]

                while len(results) < self.elements_per_iter[pos]:
                    try:
                        results.append(next(itr))
                    except StopIteration:
                        itr = itrs[pos] = self.stores[pos].iterate(cyclic=cyclic)
                        has_done_epoch[pos] = True

                yield tuple(results)

    def concat(self, other, elements_per_iter):
        self.stores.append(other)
        self.elements_per_iter.append(elements_per_iter)

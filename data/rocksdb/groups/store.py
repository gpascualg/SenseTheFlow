import numpy as np

from .concat import RocksConcat


class RocksStore(object):
    def __init__(self):
        self.dbs = []

    def add(self, db):
        self.dbs.append(db)

    def split(self, num_samples):
        splits = [db.split(num_samples) for db in self.dbs]
        
        split_a = RocksStore()
        split_b = RocksStore()

        for db_a, db_b in splits:
            split_a.add(db_a)
            split_b.add(db_b)

        return split_a, split_b


    def concat(self, other, elements_per_iter_self=1, elements_per_iter_other=1):
        return RocksConcat([self, other], [elements_per_iter_self, elements_per_iter_other])

    def iterate(self):
        itrs = [db.iterate() for db in self.dbs]

        while True:
            yield tuple([next(itr) for itr in itrs])

    def close(self):
        for db in self.dbs:
            db.close()

    def close_iterator(self):
        for db in self.dbs:
            db.close_iterator()

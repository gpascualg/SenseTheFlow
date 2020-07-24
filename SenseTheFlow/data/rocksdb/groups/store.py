import numpy as np

from .concat import RocksConcat
from ..loaders import wildcard


class RocksStore(object):
    def __init__(self):
        self.dbs = []

    def add(self, db):
        self.dbs.append(db)

    def insert(self, db, position):
        self.dbs.insert(position, db)

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

    def __add__(self, other):
        if isinstance(other, wildcard.RocksWildcard):
            self.add(other)
            return self

        elif isinstance(other, RocksStore):
            for db in other.dbs:
                self.add(db)
                return self
        
        else:
            raise RuntimeError("Unexpected input")

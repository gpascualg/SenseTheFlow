from scipy import misc
import os
import sys
import time
import random
import numpy as np
from tqdm import tqdm
from Queue import Queue
from threading import Thread, Lock, BoundedSemaphore
from itertools import cycle
from cv2 import resize

# Base class to load data in batches
class DataLoader(object):
    def __init__(self, file_path, data_path='./', mean=None, resize=None, crop=None,
                 scale=1.0, batch_size=1, prefetch=0, start_prefetch=True, **kwargs):
        self.file_path = file_path
        self.data_path = data_path
        self.mean = mean
        self.resize = resize
        self.crop = crop
        self.scale = scale
        self.current = 0

        self.batch_size = batch_size
        self.prefetch = prefetch
        self.prefetch_started = False
        self.prefetch_stop = False
        self.prefetch_queue = Queue(maxsize=prefetch)

        if start_prefetch:
            self.start_prefetch()

    # Starts the prefetching thread
    def start_prefetch(self):
        if self.prefetch > 0 and not self.prefetch_started:
            self.prefetch_thread = Thread(target=self.batch_prefetcher, args=(self,))
            self.prefetch_thread.start()
            self.prefetch_started = True

        return self

    # Waits until all the batches are prefetched
    def wait_prefetch(self):
        if self.prefetch > 0:
            last = 0
            print >> sys.stdout, "Prefetching data"

            # Show loading bar
            with tqdm(total=self.prefetch) as pbar:
                # Wait until completely done
                while self.prefetch_queue.qsize() < self.prefetch:
                    current = self.prefetch_queue.qsize()
                    if current - last > 0:
                        pbar.update(current - last)
                        last = current

                    time.sleep(0.200)

                # Update in case the last was not taken into account
                if last < self.prefetch_queue.qsize():
                    pbar.update(self.prefetch_queue.qsize() - last)

        time.sleep(0.5)
        return self

    # Function ran by the thread
    def batch_prefetcher(x, self):
        while not self.prefetch_stop:
            # Generate a batch
            batch = self.gen_batch()

            # Keep looping until it is inserted
            while not self.prefetch_stop:
                try:
                    # Insert batch, raise exception if not inserted by 1 sec.
                    self.prefetch_queue.put(batch, True, 1.0)
                    break
                except:
                    pass

    # Returns a batch
    def next_batch(self):
        if self.prefetch > 0:
            # It is prefetching, get one from the queue
            batch = self.prefetch_queue.get()
        else:
            # Not prefetching, directly generate one
            batch = self.gen_batch()

        return batch

    # Reorders the list
    def shuffle(self, a, b):
        if len(a) != len(b):
            print >> sys.stderr, "Unmatching sets"

        combined = zip(a, b)
        random.shuffle(combined)
        return zip(*combined)

    # Processes an image
    def process_image(self, img):
        # Convert to float and scale
        img = img.astype(np.float32) * self.scale

        # Resize if necessary
        if self.resize is not None:
            img = resize(img, (self.resize[0], self.resize[1]))

        # Crop if necessary
        if self.crop is not None:
            h, w = img.shape[0], img.shape[1]
            crop_height, crop_width = ((h-self.crop[0])/2, (w-self.crop[1])/2)
            img = img[crop_height:crop_height+self.crop[0], crop_width:crop_width+self.crop[1], ...]

        # Substract mean if necessary
        if self.mean is not None:
            img -= self.mean

        return img

    def reset(self):
        pass

    def close(self):
        if self.prefetch > 0:
            self.prefetch_stop = True
            self.prefetch_thread.join()


# This class loads data from a .txt file, where each line is the
# combination of "path class"
class TextLoader(DataLoader):
    def __init__(self, shuffle=True, **kwargs):
        super(TextLoader, self).__init__(**kwargs)

        # Open file and read all lines
        lines = open(self.file_path).readlines()
        pairs = [line.split() for line in lines]

        # Split paths and classes/labels
        paths, labels = zip(*pairs)
        
        self.image_filter = None
        if 'image_filter' in kwargs:
            self.image_filter = kwargs['image_filter']

        # Shuffle if we have to
        if shuffle:
            self.bare_paths, self.bare_labels = self.shuffle(paths, labels)
        else:
            self.bare_paths, self.bare_labels = paths, labels

        # Assert batch size is less than the total data size
        assert self.batch_size <= len(self.bare_paths), "Length Batch " + str(len(self.bare_paths)) + " > " + str(len(self.bare_paths))

        # Make batches cyclic, so it automatically restarts
        self.paths, self.labels = cycle(self.bare_paths), cycle(self.bare_labels)

    # Generate a batch by simply processing K images and labels
    def gen_batch(self):
        images = []
        labels = []
        
        while len(images) < self.batch_size:
            try:
                label = next(self.labels)
                image = self.process_image(misc.imread(self.data_path + next(self.paths)))
            except:
                continue
                
            if self.image_filter is None or self.image_filter(image):
                images.append(image)
                labels.append(label)
          
        return np.asarray(images), np.reshape(np.asarray(labels), (-1, 1))

    # Restart
    def reset(self):
        self.paths, self.labels = cycle(self.bare_paths), cycle(self.bare_labels)

    def __len__(self):
        return len(self.labels)


# Simply warps a numpy array to give the cyclic behaviour
class NumpyLoader(DataLoader):
    def __init__(self, **kwargs):
        super(NumpyLoader, self).__init__(**kwargs)
        self.data = cycle(self.file_path)

        assert self.batch_size <= len(self.data), "Length Batch " + str(n) + " > " + str(len(self.data))

    def gen_batch(self):
        data = [next(self.data) for i in range(self.batch_size)]
        return data, None

    def reset(self):
        self.data = cycle(self.file_path)

    def __len__(self):
        return len(self.labels)


# Lmdb loading is only available when Caffe is installed
try:
    import lmdb

    caffe_root = '/home/deep/caffe'
    sys.path.insert(0, caffe_root + '/python')

    import caffe
    from caffe.proto import caffe_pb2

    # Lmdb loader class
    class LmdbLoader(DataLoader):
        def __init__(self, **kwargs):
            # Open the databse
            self.env = lmdb.open(kwargs['file_path'])
            self.txn = lmdb.Transaction(self.env)
            self.cur = self.txn.cursor()
            self.num_items = self.env.stat()['entries']
            self.labels = None

            super(LmdbLoader, self).__init__(**kwargs)

            assert self.batch_size < self.num_items, "Length Batch " + str(n) + " > " + str(self.num_items)

        # The labels might be another LMDB
        def load_labels(self, **kwargs):
            kwargs['batch_size'] = self.batch_size
            kwargs['prefetch'] = 0 # We already prefetch here!
            self.labels = LmdbLoader(**kwargs)

        def read_image(self, path):
            img = misc.imread(self.data_path + path)
            return self.process_image(img)

        # Returns an image from the database
        def get_one(self):
            # Next item
            if not self.cur.next():
                self.cur.first()

            # Get the item
            key, value = self.cur.item()

            # Generate the Caffe datum and deserialize the item
            datum = caffe_pb2.Datum()
            datum.ParseFromString(value)
            data = caffe.io.datum_to_array(datum)

            # CxHxW to HxWxC (OpenCV)
            img = np.transpose(data, (1,2,0))
            img = self.process_image(img)
            return img, key

        # Generates a batch
        def gen_batch(self):
            pairs = [self.get_one() for i in range(self.batch_size)]
            images, labels = zip(*pairs)
            images = np.asarray(images)

            if self.labels is None:
                # Tensorflow labels are 2D
                labels = np.reshape(np.asarray(labels), (-1, 1))
            else:
                labels, _ = self.labels.next_batch()

            return images, labels

        # Closes the database
        def close(self):
            super(LmdbLoader, self).close()
            self.cur.close()
            self.env.close()

            if self.labels is not None:
                self.labels.close()

            return self

        def reset(self):
            self.cur.first()

            if self.labels is not None:
                self.labels.reset()

        def __len__(self):
            return self.num_items
except:
    print >> sys.stderr, "LMDB Loading won't be available"


try:
    import rocksdb
    
    # Lmdb loader class
    class RocksLoader(DataLoader):
        def __init__(self, **kwargs):
            # Open the databse
            kwargs['dbtype'] = 'float' if 'dbtype' not in kwargs else kwargs['dbtype']

            self.dbname = kwargs['file_path']
            self.max_key_size = kwargs['max_key_size']
            self.dbtype = kwargs['dbtype']
            self.itr_size = kwargs['iterator_size']
            self.db = rocksdb.RocksStore(self.dbname, max_key_size=self.max_key_size, dtype=self.dbtype)
            self.itr = self.db.iterate(self.itr_size)

            super(RocksLoader, self).__init__(**kwargs)


        # Returns an image from the database
        def get_one(self):
            # Next item
            return next(self.itr)

        # Generates a batch
        def gen_batch(self):
            pairs = [self.get_one() for i in range(self.batch_size)]
            images, labels = zip(*pairs)
            images = np.asarray(images)
            labels = np.reshape(np.asarray(labels), (-1, 1))

            return images, labels

        # Closes the database
        def close(self):
            super(RocksLoader, self).close()
            self.itr.close()
            self.db.close()
            
            return self

        def reset(self):
            self.itr = self.db.iterate(self.itr_size)

        def __len__(self):
            return 0
except Exception as e:
    print >> sys.stderr, "RocksDB Loading won't be available"
    raise e
    
    
# Merges two dicts into a single one
def merge_two_dicts(x, y):
    '''Given two dicts, merge them into a new dict as a shallow copy.'''
    z = x.copy()
    z.update(y)
    return z


# Default to LmdbLoader is possible, otherwise use the TextLoader
try:
    DefaultLoader = LmdbLoader
except:
    DefaultLoader = TextLoader

# Warps train and test in a single class
class Dataset(object):
    def __init__(self, config={}, dtype=DefaultLoader):
        self.train = None
        self.test = None

        # If there is any configuration
        if len(config) > 0:
            # Get the common parameters
            common = {} if 'common' not in config else config['common']

            # Merge common with train, if available, and create the DataLoader
            if 'train' in config:
                train = merge_two_dicts(config['train'], common)
                self.load_train(dtype=dtype, **train)

            # Merge common with test, if available, and create the DataLoader
            if 'test' in config:
                test = merge_two_dicts(config['test'], common)
                self.load_test(dtype=dtype, **test)

            # Save batch size just in case there is no train/test specified
            # Defaults to 0
            self.__batch_size = 0
            if 'common' in config and 'batch_size' in config['common']:
                self.__batch_size = config['common']['batch_size']

    # Returns the batch size, train over test over default
    def batch_size(self):
        if self.train is not None:
            return self.train.batch_size

        if self.test is not None:
            return self.test.batch_size

        return self.__batch_size

    # Postponed train loading
    def load_train(self, dtype=TextLoader, **kwargs):
        self.train = dtype(**kwargs)

    # Postponed test loading
    def load_test(self, dtype=TextLoader, **kwargs):
        self.test = dtype(**kwargs)

    def close(self):
        if self.train is not None:
            self.train.close()

        if self.test is not None:
            self.test.close()

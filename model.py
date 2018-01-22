import types
import tensorflow as tf
import numpy as np
from .config import bar
from heapq import heappush, heappop
import os

try:
    from inspect import signature
except:
    from funcsigs import signature


class Model(object):
    current = None
    
    def __init__(self, parser_fn, model_fn, generator, batch_size, config=None):
        self.__session = tf.Session(config=config)
        
        self.__parser_fn = parser_fn
        self.__model_fn = model_fn

        self.__data_generator = generator
        self.__batch_size = batch_size
        
        self.__epoch = 0
        self.__init = None
        self.__feed_dict = None
        self.__last_feed_dict = None
        self.__results = None
        self.__bar = None
        
        self.__callbacks = []
        
    def add_callback(self, steps, func):
        assert type(func) == types.FunctionType, "Expected a function in func"
        assert len(signature(func).parameters) == 1, "Expected func to have 1 parameters (model)"
        
        self.__callbacks.append((steps, func))

    def input_fn(self, is_training, generator, batch_size,
                 parser_fn, shuffle_buffer=64, 
                 num_parallel_calls=5, num_epochs=1):
        dataset = tf.dataset.Dataset.from_generator(*generator)

        # Pre-parsing
        if is_training:
            dataset.shuffle(buffer_size=shuffle_buffer)

        dataset = dataset.map(lambda val: parser_fn(val, is_training), num_parallel_calls=num_parallel_calls)
        dataset = dataset.prefetch(batch_size)

        # Post-parsing
        if is_training:
            dataset.shuffle(buffer_size=shuffle_buffer)

        dataset = dataset.repeat(num_epochs)
        dataset = dataset.batch(batch_size)
        
        iterator = dataset.make_one_shot_iterator()
        feats, labels = iterator.get_next()
        return feats, labels

    # Getters
    def epoch(self):
        return self.__epoch
    
    def session(self):
        return self.__session

    def bar(self):
        return self.__bar
    
        
    def __enter__(self):
        Model.current = self
        return self

    def __exit__(self, type, value, tb):
        self.__session.close()
        Model.current = None       


    def train(self, model_dir, epochs, epochs_per_eval):
        os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
  
        # Set up a RunConfig to only save checkpoints once per training cycle.
        run_config = tf.estimator.RunConfig().replace(save_checkpoints_secs=1e9)

        classifier = tf.estimator.Estimator(
            model_fn=self.__model_fn, model_dir=model_dir, config=run_config,
            params={})

        self.__bar = bar(range(epochs // epochs_per_eval))
        for self.__epoch in self.__bar:
            # logging_hook = tf.train.LoggingTensorHook(
            #     tensors={}, every_n_iter=100)

            classifier.train(
                input_fn=lambda: self.input_fn(True, self.__data_generator, self.__batch_size,
                self.__parser_fn, shuffle_buffer=64, 
                num_parallel_calls=5, num_epochs=1)
            )
        

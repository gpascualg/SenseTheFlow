import types
import tensorflow as tf
import numpy as np
from .config import bar
from heapq import heappush, heappop

try:
    from inspect import signature
except:
    from funcsigs import signature


class Model(object):
    current = None
    
    def __init__(self, config=None):
        self.__session = tf.Session(config=config)
        
        self.__init = None
        self.__feed_dict = None
        self.__last_feed_dict = None
        self.__results = None
        
        self.__callbacks = []
        
    def add_callback(self, steps, func):
        assert type(func) == types.FunctionType, "Expected a function in func"
        assert len(signature(func).parameters) == 1, "Expected func to have 1 parameters (model)"
        
        self.__callbacks.append((steps, func))
           
    def set_feed_dict(self, func):
        assert type(func) == types.FunctionType, "Expected a function in func"
        assert len(signature(func).parameters) == 1, "Expected func to have 1 parameters (model)"
        
        self.__feed_dict = func
        
    def set_init(self, func):
        assert type(func) == types.FunctionType, "Expected a function in func"
        assert len(signature(func).parameters) == 1, "Expected func to have 1 parameters (model)"
        
        self.__init = func
        
        
    # Getters
    def feed_dict(self):
        return self.__last_feed_dict
    
    def step(self):
        return self.__step
    
    def session(self):
        return self.__session

    def results(self):
        return self.__results

    def bar(self):
        return self.__bar
    
        
    def __enter__(self):
        Model.current = self
        return self

    def __exit__(self, type, value, tb):
        self.__session.close()
        Model.current = None       


    def train(self, train_op, steps):
        assert type(self.__feed_dict) == types.FunctionType, "Expected a function in feed_dict, call set_feed_dict"
        
        if self.__init is not None:
            self.__init(self)
        else:
            self.__session.run(tf.global_variables_initializer())
        
        self.__bar = bar(range(steps))
        for self.__step in self.__bar:
            self.__last_feed_dict = self.__feed_dict(self)
            self.__results = self.__session.run(train_op, self.__last_feed_dict)
            
            for steps, func in self.__callbacks:
                if self.__step % steps == 0:
                    func(self)
        

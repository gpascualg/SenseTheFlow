import types
import tensorflow as tf
import numpy as np
from tqdm import tqdm

try:
    from inspect import signature
except:
    from funcsigs import signature


class Layer(object):
    def __init__(self, name, activation):
        self.name = name
        self.verbose = False
        self.trainable = True
        self.wd = None
        self.variables = {}
        self.activation = activation if activation is not None else lambda x: x
        
    
    def __call__(self, layer):
        pass
    
    
    def make_var(self, name, shape, initializer=None):
        # Output some information if verbose is set
        if self.verbose:
            scope = tf.get_variable_scope()
            activation = None if self.activation is None else self.activation.__name__
            info_a = "{}/{}:".format(scope.name, name)
            info_b = "[Trainable: {}\tActivation: {}]".format(self.trainable, activation)
            print(info_a + " " * (80 - len(info_a)) + info_b)

        # Create the variable
        var = tf.get_variable(name, shape, trainable=self.trainable, initializer=initializer, dtype=np.float32)
        self.variables[name] = var

        # Apply weight decay (regularization) if specified
        if self.wd is not None and self.trainable:
            weight_decay = tf.mul(tf.nn.l2_loss(var), self.wd, name='weight_loss')
            tf.add_to_collection('losses', weight_decay)

        return var
    
    
class Convolution(Layer):
    def __init__(self, name, shape, output_size, stride, padding='SAME', groups=1, activation=tf.nn.relu):
        super(Convolution, self).__init__(name, activation)
        
        # Save some variables
        self.shape = shape
        self.output_size = output_size
        self.stride = stride
        self.padding = padding
        self.groups = groups

    def __call__(self, input):        
        # Get number of input channels
        input_channels = int(input.get_shape()[-1])
        
        # Create variables
        with tf.variable_scope(self.name) as scope:
            self.weights = self.make_var(name='weights', shape=[self.shape[0], self.shape[1], 
                                                                input_channels/self.groups, 
                                                                self.output_size])
            
            self.biases = self.make_var(name='biases', shape=[self.output_size])
            
        # Create lambda function for the convolution
        convolve = lambda i, k: tf.nn.conv2d(i, k, strides = [1, self.stride[0], self.stride[1], 1],
                                             padding=self.padding)
        
        if self.groups == 1:
            conv = convolve(input, self.weights)

        # In the cases of multiple groups, split inputs & weights and
        else:
            # Split input and weights and convolve them separately
            input_groups = tf.split(axis=3, num_or_size_splits=self.groups, value=input)
            weight_groups = tf.split(axis=3, num_or_size_splits=self.groups, value=self.weights)
            output_groups = [convolve(i, k) for i,k in zip(input_groups, weight_groups)]

            # Concat the convolved output together again
            conv = tf.concat(axis=3, values=output_groups)

        # Add biases
        bias = tf.reshape(tf.nn.bias_add(conv, self.biases), conv.get_shape().as_list())

        # Apply relu function
        relu = self.activation(bias, name=self.name)

        return relu

class Dense(Layer):
    def __init__(self, name, output_size, activation=tf.nn.relu):
        super(Dense, self).__init__(name, activation)

        self.output_size = output_size

        
    def __call__(self, input):
        input_size = input.get_shape().as_list()[1]
        shape = [input_size, self.output_size]

        with tf.variable_scope(self.name):
            weights = self.make_var('weights', shape=shape)
            biases = self.make_var('biases', shape=[self.output_size])

        return self.activation(tf.nn.xw_plus_b(input, weights, biases))
    

class Cosine(Dense):
    def __init__(self, name, output_size, activation=tf.nn.relu):
        super(Cosine, self).__init__(name, activation)

        self.output_size = output_size

        
    def __call__(self, input):
        input_size = input.get_shape().as_list()[1]
        shape = [input_size, self.output_size]

        with tf.variable_scope(self.name):
            weights = self.make_var('weights', shape=shape)
            biases = self.make_var('biases', shape=[self.output_size])

        w_norm = tf.sqrt(tf.reduce_sum(weights**2, axis=0, keep_dims=True) + biases**2)
        x_norm = tf.sqrt(tf.reduce_sum(input**2, axis=1, keep_dims=True))
        wx_normalized = (tf.matmul(input, weights) + biases) / tf.maximum(1e-6, w_norm * x_norm)
        return self.activation(wx_normalized)
    
    
class MaxPool(Layer):
    def __init__(self, name, shape, stride, padding='SAME'):
        super(MaxPool, self).__init__(name, None)
        
        self.shape = shape
        self.stride = stride
        self.padding = padding
        
    
    def __call__(self, input):
        return tf.nn.max_pool(input, ksize=[1, self.shape[0], self.shape[1], 1],
                              strides=[1, self.stride[0], self.stride[1], 1],
                              padding=self.padding, name=self.name)

    
class LRN(Layer):
    def __init__(self, name, radius, alpha, beta, bias=1.0):
        super(LRN, self).__init__(name, None)
        
        self.radius = radius
        self.alpha = alpha
        self.beta = beta
        self.bias = bias
        
    
    def __call__(self, input):
        return tf.nn.local_response_normalization(input, depth_radius=self.radius,
                                                  alpha=self.alpha, beta=self.beta,
                                                  bias=self.bias, name=self.name)
    
    
class Dropout(Layer):
    def __init__(self, name, keep_prob):
        super(Dropout, self).__init__(name, None)
        
        self.keep_prob = keep_prob
    
    
    def __call__(self, input):
        return tf.nn.dropout(input, keep_prob=self.keep_prob, name=self.name)
    
    

class BatchNormalization(Layer):
    def __init__(self, name, is_training, decay=0.999, epsilon=1e-3):
        super(BatchNormalization, self).__init__(name, None)
        
        self.is_training = is_training
        self.decay = decay
        self.epsilon = epsilon
        
        
    def __call__(self, input):
        def _inner(input, shape):
            input = tf.reshape(input, (int(shape[0]), -1))
            scale = tf.Variable(tf.ones([input.get_shape()[-1]]))
            beta = tf.Variable(tf.zeros([input.get_shape()[-1]]))
            pop_mean = tf.Variable(tf.zeros([input.get_shape()[-1]]), trainable=False)
            pop_var = tf.Variable(tf.ones([input.get_shape()[-1]]), trainable=False)

            def update_training():
                batch_mean, batch_var = tf.nn.moments(input,[0])
                train_mean = tf.assign(pop_mean,
                                       pop_mean * self.decay + batch_mean * (1 - self.decay))
                train_var = tf.assign(pop_var,
                                      pop_var * self.decay + batch_var * (1 - self.decay))

                with tf.control_dependencies([train_mean, train_var]):
                    return tf.nn.batch_normalization(input,
                        batch_mean, batch_var, beta, scale, self.epsilon)
            
            def update_test():
                return tf.nn.batch_normalization(input,
                    pop_mean, pop_var, beta, scale, self.epsilon)
            
            return tf.cond(self.is_training, update_training, update_test)
            
        shape = input.get_shape()
        out = _inner(input, shape)
        return tf.reshape(out, shape)



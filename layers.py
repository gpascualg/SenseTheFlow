from decorator import decorator
import random
import numpy as np
from scipy import stats

import tensorflow as tf


# Common method which parses input parameters and sets them as class
# attributes, works for all names in Layer.Defaults
@decorator
def parse_config(f, *args, **kwargs):
    self = args[0]

    for key, value in kwargs.iteritems():
        if not key in Layer.Defaults:
            raise Exception('Unexpected key `{}`'.format(key))

    for key, value in Layer.Defaults.iteritems():
        if not key in kwargs: #or kwargs[key] is None:
            kwargs[key] = value

    for key, value in kwargs.iteritems():
        setattr(self, key, value)

    return f(*args, **kwargs)


# Base class for layers
class Layer(object):
    Defaults = {
        'verbose': False,
        'trainable': True,
        'wd': 0.0005,
        'padding': 'SAME',
        'activation': tf.nn.relu
    }

    @parse_config
    def __init__(self, name, **config):
        self.name = name
        self.variables = []

    # Creates a variable
    def make_var(self, name, shape):
        # Output some information if verbose is set
        if self.verbose:
            scope = tf.get_variable_scope()
            activation = None if self.activation is None else self.activation.__name__
            info_a = "{}/{}:".format(scope.name, name)
            info_b = "[Trainable: {}\tActivation: {}]".format(self.trainable, activation)
            print info_a + " " * (80 - len(info_a)) + info_b

        # Create the variable
        var = tf.get_variable(name, shape, trainable=self.trainable)
        self.variables.append(var)

        # Apply weight decay (regularization) if specified
        if self.wd is not None and self.trainable:
            weight_decay = tf.mul(tf.nn.l2_loss(var), self.wd, name='weight_loss')
            tf.add_to_collection('losses', weight_decay)

        return var

# Convolutional layer
class Conv(Layer):
    def __init__(self, name, kernel_size, output_size, stride_size=(1, 1), group=1, **config):
        assert output_size%group==0

        super(Conv, self).__init__(name, **config)

        self.kernel_size = kernel_size
        self.output_size = output_size
        self.stride_size = stride_size
        self.group = group

    def __call__(self, x):
        # Assert it is multiple of the convolve group
        self.c_i = x.get_shape()[-1]
        assert self.c_i%self.group==0

        # Convolve function
        convolve = lambda i, k: tf.nn.conv2d(i, k, [1, self.stride_size[0], self.stride_size[1], 1], padding=self.padding)

        # Create the kernel and biases in a scope
        with tf.variable_scope(self.name) as scope:
            kernel = self.make_var('weights', shape=[self.kernel_size[0], self.kernel_size[1],
                                                     self.c_i/self.group, self.output_size])
            biases = self.make_var('biases', shape=[self.output_size])
            
            pre_call = getattr(self, "pre_call", None)
            if callable(pre_call):
                kernel, biases = pre_call(kernel, biases)

            # If the are no groups, simply convolve, else, split by groups and convolve
            if self.group == 1:
                conv = convolve(x, kernel)
            else:
                input_groups = tf.split(3, group, x)
                kernel_groups = tf.split(3, group, kernel)
                output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
                conv = tf.concat(3, output_groups)

            # Apply non-linearities if any
            if self.activation is not None:
                bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())
                return self.activation(bias, name=scope.name)

            # Add bias and return
            return tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list(), name=scope.name)
        

MASK_TYPE_A = "A"
MASK_TYPE_B = "B"
class MaskedConv(Conv):
    def __init__(self, name, kernel_size, output_size, mask_type=None, stride_size=(1, 1), group=1, **config):
        super(MaskedConv, self).__init__(name, kernel_size, output_size, stride_size, group, **config)
        
        self.mask_type = mask_type
        
    def pre_call(self, kernel, biases):        
        if self.mask_type is not None:
            mask = np.ones((self.kernel_size[0], self.kernel_size[1], self.c_i/self.group, self.output_size))
            
            center_h = int(self.kernel_size[0] / 2)
            center_w = int(self.kernel_size[1] / 2)
            
            for i in xrange(self.kernel_size[1]):
                for j in xrange(self.kernel_size[0]):
                    if (j > center_h) or (j == center_h and i > center_w):
                        mask[j, i, :, :] = 0.
            
            N_CHANNELS=3 # pass by param
            for i in xrange(N_CHANNELS):
                for j in xrange(N_CHANNELS):
                    if (self.mask_type == MASK_TYPE_A and i >= j) or (self.mask_type == MASK_TYPE_B and i > j):
                        mask[center_h, center_w, j::N_CHANNELS, i::N_CHANNELS] = 0

            kernel *= tf.constant(mask, dtype=tf.float32)
            tf.add_to_collection('masked_conv_%s_weights' % self.mask_type, kernel)
                
        return kernel, biases


# Max pooling layer
class MaxPool(Layer):
    def __init__(self, name, kernel_size, stride_size=(2,2), **config):
        super(MaxPool, self).__init__(name, **config)

        self.kernel_size = kernel_size
        self.stride_size = stride_size

    def __call__(self, x):
        return tf.nn.max_pool(x,
                              ksize=[1, self.kernel_size[0], self.kernel_size[1], 1],
                              strides=[1, self.stride_size[0], self.stride_size[1], 1],
                              padding=self.padding,
                              name=self.name)


# Fully connected layer (Dense Layer)
class Dense(Layer):
    def __init__(self, name, output_size, **config):
        super(Dense, self).__init__(name, **config)

        self.output_size = output_size

    def __call__(self, x):
        input_size = x.get_shape().as_list()[1]
        shape = [input_size, self.output_size]

        with tf.variable_scope(self.name, initializer=tf.truncated_normal(shape, stddev=0.1)):
            weights = self.make_var('weights', shape=None)

        with tf.variable_scope(self.name, initializer=tf.constant(0.1, shape=[self.output_size])):
            biases = self.make_var('biases', shape=None)

        op = self.activation if self.activation is not None else lambda x: x
        return op(tf.nn.xw_plus_b(x, weights, biases))


class BatchNorm(Layer):
    def __init__(self, name, output_size, phase_train, **config):
        super(BatchNorm, self).__init__(name, **config)

        self.output_size = output_size
        self.phase_train = phase_train

    def __call__(self, x):
        with tf.variable_scope(self.name, initializer=tf.constant(0.0, shape=[self.output_size])):
            beta = self.make_var('beta', shape=None)

        with tf.variable_scope(self.name, initializer=tf.constant(1.0, shape=[self.output_size])):
            gamma = self.make_var('gamma', shape=None)

            batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
            ema = tf.train.ExponentialMovingAverage(decay=0.5)

            def mean_var_with_update():
                ema_apply_op = ema.apply([batch_mean, batch_var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(batch_mean), tf.identity(batch_var)

            mean, var = tf.cond(self.phase_train,
                                mean_var_with_update,
                                lambda: (ema.average(batch_mean), ema.average(batch_var)))
            normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3, name='batch_norm')

        return normed


class MixtureDensity(Layer):
    def __init__(self, name, k, **config):
        super(MixtureDensity, self).__init__(name, **config)

        self.k = k

    def __call__(self, x, y):
        with tf.variable_scope(self.name):
            mus = Dense(name='mus', output_size=self.k)
            sigmas = Dense(name='sigmas', output_size=self.k, activation=tf.exp)
            pi = Dense(name='pi', output_size=self.k, activation=tf.nn.softmax)

            self.mus = mus(x)
            self.sigmas = sigmas(x)
            self.pi = pi(x)

            self.variables += mus.variables + sigmas.variables + pi.variables

            return self.log_prob(y, self.pi, self.sigmas, self.mus)

    def normal(self, y, mu, sigma):
        result = tf.sub(y, mu)
        result = tf.mul(result,tf.inv(sigma))
        result = -tf.square(result)/2
        return tf.mul(tf.exp(result), tf.inv(sigma)) * (1 / np.sqrt(2*np.pi))

    def log_prob(self, out_pi, out_sigma, out_mu, y):
        result = self.normal(y, out_mu, out_sigma)
        result = tf.mul(result, out_pi)
        result = tf.reduce_sum(result, 1, keep_dims=True)
        result = -tf.log(tf.maximum(result, 1e-20))
        return tf.reduce_mean(result)

    def sample_from_mixture(self, pred_weights, pred_means, pred_std, amount):
        """
        Draws samples from mixture model.
        Returns 2 d array with input X and sample from prediction of Mixture Model
        """
        samples = np.zeros((amount,))
        n_mix = len(pred_weights[0])
        to_choose_from = np.arange(n_mix)
        for j,(weights, means, std_devs) in enumerate(zip(pred_weights, pred_means, pred_std)):
            index = np.random.choice(to_choose_from, p=weights)
            samples[j]= stats.norm.rvs(means[index], std_devs[index], size=1)
            if j == amount -1:
                break

        return samples


class StableMixtureDensity(Layer):
    def __init__(self, name, k, **config):
        super(StableMixtureDensity, self).__init__(name, **config)

        self.k = k

    def __call__(self, x, y):
        with tf.variable_scope(self.name):
            mus = Dense(name='mus', output_size=self.k)
            sigmas = Dense(name='sigmas', output_size=self.k, activation=tf.exp)
            pi = Dense(name='pi', output_size=self.k, activation=tf.nn.softmax)

            self.mus = mus(x)
            self.sigmas = sigmas(x)
            self.pi = pi(x)

            self.variables += mus.variables + sigmas.variables + pi.variables

            for x in [self.pi, self.mus, self.sigmas]:
                print '{}: {}'.format(x.name, x.get_shape().as_list())

            return self.log_prob(y, self.pi, self.sigmas, self.mus)

    def logpdf(self, x, loc, scale):
        z = (x - loc) / scale
        return (-0.5*tf.log(2*np.pi) - tf.log(scale) - 0.5*tf.square(z))

    def log_prob(self, out_pi, out_sigma, out_mu, y):
        z = self.logpdf(y, out_mu, out_sigma)
        result = tf.exp(z)
        result = tf.mul(result, out_pi)
        result = tf.reduce_sum(result, 1, keep_dims=True)
        result = -tf.log(tf.maximum(result, 1e-20))
        return tf.reduce_mean(result)

    def sample_from_mixture(self, pred_weights, pred_means, pred_std, amount):
        def get_pi_idx(x, pdf):
            N = pdf.size
            accumulate = 0
            for i in range(0, N):
                accumulate += pdf[i]
                if (accumulate >= x):
                    return i

            return -1

        def generate_ensemble(out_pi, out_mu, out_sigma, M = 1):
            result = np.random.rand(amount, M) # initially random [0, 1]
            rn = np.random.randn(amount, M) # normal random matrix (0.0, 1.0)
            mu = 0
            std = 0
            idx = 0

            # transforms result into random ensembles
            for j in range(0, M):
                for i in range(0, amount):
                    idx = get_pi_idx(result[i, j], out_pi[0,:])
                    mu = out_mu[j, idx]
                    std = out_sigma[j, idx]
                    result[i, j] = mu + rn[i, j] * std
            return result[:,0]

        return generate_ensemble(pred_weights, pred_means, pred_std)
        #
        # Other method
        samples = np.zeros((amount,))
        n_mix = len(pred_weights[0,:])
        to_choose_from = np.arange(n_mix)
        weights = pred_weights[0,:]
        means = pred_means[0,:]
        std_devs = pred_std[0,:]

        for j in range(amount):
            index = np.random.choice(to_choose_from, p=weights)
            samples[j] = stats.norm.rvs(means[index], std_devs[index], size=1)

        return samples


# Based on http://arxiv.org/abs/1308.0850
class MixtureDensity2D(Layer):
    def __init__(self, name, k, input_tensors=None, **config):
        super(MixtureDensity2D, self).__init__(name, **config)

        self.k = k
        self.input_tensors = input_tensors

    def __call__(self, x, y1, y2):
        with tf.variable_scope(self.name):
            """
            pi = Dense(name='pi', output_size=self.k, activation=tf.nn.softmax)
            mus1 = Dense(name='mus1', output_size=self.k, activation=tf.tanh)
            mus2 = Dense(name='mus2', output_size=self.k, activation=tf.tanh)
            sigmas1 = Dense(name='sigmas1', output_size=self.k, activation=tf.exp)
            sigmas2 = Dense(name='sigmas2', output_size=self.k, activation=tf.exp)
            corr = Dense(name='corr', output_size=self.k, activation=tf.tanh)

            # Encode X as a (256*256) vector of row/col indices instead of (256*256*2) ab values
            # Normalize to range [0,1] each component, and to [-0.5,0.5] the final x
            #x = tf.mul(x1, x2)

            self.pi = pi(x)
            self.mus1 = mus1(x)
            self.mus2 = mus2(x)
            self.sigmas1 = sigmas1(x)
            self.sigmas2 = sigmas2(x)
            self.corr = corr(x)


            for x in [pi, mus1, mus2, sigmas1, sigmas2, corr]:
                self.variables += x.variables
            """

            # Build the tensors
            if self.input_tensors is None:
                outputs = Dense(name='pi', output_size=self.k * 6, activation=tf.tanh)
                pi, self.mus1, self.mus2, sigmas1, sigmas2, self.corr = tf.split(1, 6, outputs(x))

                # softmax all the pi's:
                max_pi = tf.reduce_max(pi, 1, keep_dims=True)
                pi = tf.sub(pi, max_pi)
                pi = tf.exp(pi)
                normalize_pi = tf.inv(tf.reduce_sum(pi, 1, keep_dims=True))
                self.pi = tf.mul(normalize_pi, pi)

                # exponentiate the sigmas
                self.sigmas1 = tf.exp(sigmas1)
                self.sigmas2 = tf.exp(sigmas2)

                self.variables += outputs.variables
            else:
                def has(x, o):
                    try:
                        return (o in x)
                    except:
                        try:
                            return hasattr(x, o)
                        except:
                            return False

                def fetch(x, o):
                    try:
                        return x[o]
                    except:
                        return getattr(x, o) # Raises error

                assert has(self.input_tensors, 'pi'), "Error, input_tensors should contain `pi`"
                assert has(self.input_tensors, 'sigmas1'), "Error, input_tensors should contain `sigmas1`"
                assert has(self.input_tensors, 'sigmas2'), "Error, input_tensors should contain `sigmas2`"
                assert has(self.input_tensors, 'mus1'), "Error, input_tensors should contain `mus1`"
                assert has(self.input_tensors, 'mus2'), "Error, input_tensors should contain `mus2`"
                assert has(self.input_tensors, 'corr'), "Error, input_tensors should contain `corr`"

                self.pi = fetch(self.input_tensors, 'pi')
                self.sigmas1 = fetch(self.input_tensors, 'sigmas1')
                self.sigmas2 = fetch(self.input_tensors, 'sigmas2')
                self.mus1 = fetch(self.input_tensors, 'mus1')
                self.mus2 = fetch(self.input_tensors, 'mus2')
                self.corr = fetch(self.input_tensors, 'corr')

                assert self.pi.get_shape().as_list()[1] == self.k, "Incompatible shapes for `pi`"
                assert self.sigmas1.get_shape().as_list()[1] == self.k, "Incompatible shapes for `sigmas1`"
                assert self.sigmas2.get_shape().as_list()[1] == self.k, "Incompatible shapes for `sigmas2`"
                assert self.mus1.get_shape().as_list()[1] == self.k, "Incompatible shapes for `mus1`"
                assert self.mus2.get_shape().as_list()[1] == self.k, "Incompatible shapes for `mus2`"
                assert self.corr.get_shape().as_list()[1] == self.k, "Incompatible shapes for `corr`"

            for x in [self.pi, self.mus1, self.mus2, self.sigmas1, self.sigmas2, self.corr]:
                print '{}: {}'.format(x.name, x.get_shape().as_list())

            self.loss = self.log_prob(self.pi, self.mus1, self.mus2, self.sigmas1, self.sigmas2, self.corr, y1, y2)
            return self.loss

    def normal(self, y1, y2, mu1, mu2, sigma1, sigma2, rho):
        norm1 = tf.sub(y1, mu1)
        norm2 = tf.sub(y2, mu2)
        s1s2 = tf.mul(sigma1, sigma2)
        z = tf.square(tf.div(norm1, sigma1)) + tf.square(tf.div(norm2, sigma2)) - 2 * tf.div(tf.mul(rho, tf.mul(norm1, norm2)), s1s2)
        negRho = 1 - tf.square(rho)
        result = tf.exp(tf.div(-z, 2 * negRho))
        denom = 2 * np.pi * tf.mul(s1s2, tf.sqrt(negRho))
        result = tf.div(result, denom)
        return result

    def log_prob(self, out_pi, out_mu1, out_mu2, out_sigma1, out_sigma2, out_corr, y1, y2):
        result = self.normal(y1, y2, out_mu1, out_mu2, out_sigma1, out_sigma2, out_corr)
        result = tf.mul(result, out_pi)
        result = tf.reduce_sum(result, 1, keep_dims=True)
        result = -tf.log(tf.maximum(result, 1e-20)) # at the beginning, some errors are exactly zero.
        return tf.reduce_sum(result)

    def sample(self, pi, mus1, mus2, sigmas1, sigmas2, corr, num=256*256):
        def get_pi_idx(x, pdf):
            N = pdf.size
            accumulate = 0
            for i in range(0, N):
                accumulate += pdf[i]
                if (accumulate >= x):
                    return i
            return -1

        def sample_gaussian_2d(mu1, mu2, s1, s2, rho):
            mean = [mu1, mu2]
            cov = [[s1*s1, rho*s1*s2], [rho*s1*s2, s2*s2]]
            x = np.random.multivariate_normal(mean, cov, 1)
            return x[0][0], x[0][1]

        samples = np.zeros((num,2))
        for i in range(0, num):
            idx = get_pi_idx(random.random(), pi[0])
            a,b = sample_gaussian_2d(mus1[0,idx], mus2[0,idx], sigmas1[0,idx], sigmas2[0,idx],
                                     corr[0,idx])
            samples[i,0] = a
            samples[i,1] = b

        return samples

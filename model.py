# Based in https://github.com/joelthchao/tensorflow-finetune-flickr-style/blob/master/model.py
import tensorflow as tf
import numpy as np
import sys
from tqdm import tqdm
from layers import *
from datetime import datetime

try:
    import IPython.display as display
except:
    pass


def vgg16_small(x):
    # Conv 1
    conv1_1 = Conv(name='conv1_1', kernel_size=(3, 3), output_size=64)(x)
    conv1_2 = Conv(name='conv1_2', kernel_size=(3, 3), output_size=64)(conv1_1)
    pool1 = MaxPool(name='pool1', kernel_size=(3,3))(conv1_2)

    #norm1 = lrn(pool1, 2, 2e-05, 0.75, name='norm1')

    # Conv 2
    conv2_1 = Conv(name='conv2_1', kernel_size=(3, 3), output_size=128)(pool1)
    conv2_2 = Conv(name='conv2_2', kernel_size=(3, 3), output_size=128)(conv2_1)
    pool2 = MaxPool(name='pool2', kernel_size=(3,3))(conv2_2)

    #norm2 = lrn(conv2, 2, 2e-05, 0.75, name='norm2')

    # Conv 3
    conv3_1 = Conv(name='conv3_1', kernel_size=(3, 3), output_size=256)(pool2)
    conv3_2 = Conv(name='conv3_2', kernel_size=(3, 3), output_size=256)(conv3_1)
    pool3 = MaxPool(name='pool3', kernel_size=(3,3))(conv3_2)

    # Conv 4
    conv4_1 = Conv(name='conv4_1', kernel_size=(3, 3), output_size=512)(pool3)
    conv4_2 = Conv(name='conv4_2', kernel_size=(3, 3), output_size=512)(conv4_1)
    pool4 = MaxPool(name='pool4', kernel_size=(3,3))(conv4_2)

    # Conv 5
    conv5_1 = Conv(name='conv5_1', kernel_size=(3, 3), output_size=512)(pool4)
    conv5_2 = Conv(name='conv5_2', kernel_size=(3, 3), output_size=512)(conv5_1)

    return conv5_2, {'conv4_2': conv4_2, 'conv4_1': conv4_1,
                     'conv3_2': conv3_2, 'conv3_1': conv3_1,
                     'conv2_2': conv2_2, 'conv2_1': conv2_1,
                     'conv1_2': conv1_2, 'conv1_1': conv1_1}

def vgg16_smallest(x):
    # Conv 1
    conv1_1 = Conv(name='conv1_1', kernel_size=(3, 3), output_size=64)(x)
    conv1_2 = Conv(name='conv1_2', kernel_size=(3, 3), output_size=64)(conv1_1)
    pool1 = MaxPool(name='pool1', kernel_size=(3,3))(conv1_2)

    #norm1 = lrn(pool1, 2, 2e-05, 0.75, name='norm1')

    # Conv 2
    conv2_1 = Conv(name='conv2_1', kernel_size=(3, 3), output_size=128)(pool1)
    conv2_2 = Conv(name='conv2_2', kernel_size=(3, 3), output_size=128)(conv2_1)
    pool2 = MaxPool(name='pool2', kernel_size=(3,3))(conv2_2)

    #norm2 = lrn(conv2, 2, 2e-05, 0.75, name='norm2')

    # Conv 3
    conv3_1 = Conv(name='conv3_1', kernel_size=(3, 3), output_size=256)(pool2)
    conv3_2 = Conv(name='conv3_2', kernel_size=(3, 3), output_size=256)(conv3_1)
    pool3 = MaxPool(name='pool3', kernel_size=(3,3))(conv3_2)

    # Conv 4
    conv4_1 = Conv(name='conv4_1', kernel_size=(3, 3), output_size=512)(pool3)
    conv4_2 = Conv(name='conv4_2', kernel_size=(3, 3), output_size=512)(conv4_1)

    return conv4_2, {'conv4_2': conv4_2, 'conv4_1': conv4_1,
                     'conv3_2': conv3_2, 'conv3_1': conv3_1,
                     'conv2_2': conv2_2, 'conv2_1': conv2_1,
                     'conv1_2': conv1_2, 'conv1_1': conv1_1}

# Applies a learning rate decay
class LearningRateDecay(object):
    def __init__(self, steps, decay, start, staircase):
        with tf.device('/cpu:0'):
            # Global step and decay itself must be on cpu
            self.step = tf.Variable(0, trainable=False)
            self.lr = tf.train.exponential_decay(start, self.step, steps, decay, staircase)


# Optimizer class, warps the tf.train.optimizer class
class Optimizer(object):
    def __init__(self, dtype, learning_rate, *args, **kwargs):
        self.lr = learning_rate
        if isinstance(learning_rate, LearningRateDecay):
            lr = learning_rate.lr
        else:
            lr = learning_rate

        # Create the real optimizer
        self.optimizer = dtype(*args, learning_rate=lr, **kwargs)
        self.clipping = []

    # Clip some variable gradient
    def clip(self, variable, clipping):
        self.clipping[variable.name] = clipping
        return self


    def minimize(self, loss, global_step=None, var_list=None):
        # Gradients and variables
        gvs = self.optimizer.compute_gradients(loss, var_list=var_list)

        # Apply all clippings if any
        def map_gvs(gv):
            if gv[1].name in self.clipping:
                return (self.clipping[gv[1].name](gv[0]), gv[1])
            return gv
        gvs = map(map_gvs, gvs)

        # Set the gradients to minimize
        return self.optimizer.apply_gradients(gvs, global_step=global_step)


# Model class
class Model(object):
    def __init__(self, device):
        assert hasattr(self, 'do_step'), "No do_test function"
        assert hasattr(self, 'do_test'), "No do_test function"
        assert hasattr(self, 'do_display'), "No do_display function"
        self.device = device

    # Date formatting
    def now(self):
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Clears the ipython cell
    def clear(self, sess):
        display.clear_output(wait=False)

    # Saves as a tensorflow checkpoint
    def save(self, sess, step, save_path):
        print >> sys.stderr, " - {} Iter {}: Saving".format(self.now(), step)
        saved_path = self.saver.save(sess, save_path, global_step=step)
        print >> sys.stderr, " - {} Iter {}: Saved as {}".format(self.now(), step, saved_path)

    # Automated training
    def train(self, optimizer, num_iters, var_list=None,
             test_step=1000, display_step=100, clear_step=5000, save_path=None, save_step=0):
        with tf.device(self.device):
            # Get the global step
            if isinstance(optimizer, Optimizer) and isinstance(optimizer.lr, LearningRateDecay):
                global_step = optimizer.lr.step
            else:
                global_step = None

            # Get the optimizer
            optimizer = optimizer.minimize(self.loss, global_step=global_step, var_list=var_list)

            # Saver (create checkpoints)
            with tf.device('/cpu:0'):
                self.saver = tf.train.Saver()

            # Launch the graph
            with tf.Session() as sess:
                print 'Init variables'
                if hasattr(self, 'initialize'):
                    self.initialize(sess)

                print 'Start training'
                # Display progress and call functions as needed
                for step in tqdm(range(1, num_iters + 1)):
                    self.do_step(sess, optimizer, step)

                    if step % clear_step == 0: self.clear(sess)
                    if step % test_step == 0: self.do_test(sess, optimizer, step)
                    if step % display_step == 0: self.do_display(sess, optimizer, step)
                    if save_path and step % save_step == 0: self.save(sess, step, save_path)

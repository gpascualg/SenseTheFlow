from ..helper import utils

import tensorflow as tf
import h5py

from tensorflow.keras import layers
from tensorflow.python.keras.saving import hdf5_format


class identity_block(tf.keras.Model):
    def __init__(self, data_format, kernel_size, filters, stage, block, trainable_bn, l2=0.001):
        super(identity_block, self).__init__(self)
        """The identity block is the block that has no conv layer at shortcut.

        # Arguments
            input_tensor: input tensor
            kernel_size: default 3, the kernel size of
                middle conv layer at main path
            filters: list of integers, the filters of 3 conv layer at main path
            stage: integer, current stage label, used for generating layer names
            block: 'a','b'..., current block label, used for generating layer names

        # Returns
            Output tensor for the block.
        """
        filters1, filters2, filters3 = filters
        if data_format == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        def get_l2():
            if l2 is not None and l2 > 1e-6:
                return tf.keras.regularizers.l2(l2)
            return None

        self.conv1 = layers.Conv2D(filters1, (1, 1), kernel_initializer='he_normal', kernel_regularizer=get_l2(), name=conv_name_base + '2a')
        self.bn1 = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a', trainable=trainable_bn)
        self.relu1 = layers.Activation('relu')

        self.conv2 = layers.Conv2D(filters2, kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=get_l2(), name=conv_name_base + '2b')
        self.bn2 = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b', trainable=trainable_bn)
        self.relu2 = layers.Activation('relu')

        self.conv3 = layers.Conv2D(filters3, (1, 1), kernel_initializer='he_normal', kernel_regularizer=get_l2(), name=conv_name_base + '2c')
        self.bn3 = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c', trainable=trainable_bn)

        self.add4 = layers.Add()
        self.relu4 = layers.Activation('relu')
        self.last = None

    def call(self, inputs, training):
        outputs = self.conv1(inputs)
        outputs = self.bn1(outputs, training=training)
        outputs = self.relu1(outputs)

        outputs = self.conv2(outputs)
        outputs = self.bn2(outputs, training=training)
        outputs = self.relu2(outputs)

        outputs = self.conv3(outputs)
        outputs = self.bn3(outputs, training=training)

        outputs = self.add4([outputs, inputs])
        self.last = outputs = self.relu4(outputs)
        return outputs


class conv_block(tf.keras.Model):
    def __init__(self, data_format, kernel_size, filters, stage, block, strides=(2, 2), trainable_bn=True, l2=0.001):
        super(conv_block, self).__init__(self)
        filters1, filters2, filters3 = filters
        if data_format == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        def get_l2():
            if l2 is not None and l2 > 1e-6:
                return tf.keras.regularizers.l2(l2)
            return None

        self.conv1 = layers.Conv2D(filters1, (1, 1), strides=strides, kernel_initializer='he_normal', kernel_regularizer=get_l2(), name=conv_name_base + '2a')
        self.bn1 = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a', trainable=trainable_bn)
        self.relu1 = layers.Activation('relu')

        self.conv2 = layers.Conv2D(filters2, kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=get_l2(), name=conv_name_base + '2b')
        self.bn2 = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b', trainable=trainable_bn)
        self.relu2 = layers.Activation('relu')

        self.conv3 = layers.Conv2D(filters3, (1, 1), kernel_initializer='he_normal', kernel_regularizer=get_l2(), name=conv_name_base + '2c')
        self.bn3 = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c', trainable=trainable_bn)

        self.conv_shortcut = layers.Conv2D(filters3, (1, 1), strides=strides, kernel_initializer='he_normal', kernel_regularizer=get_l2(), name=conv_name_base + '1')
        self.bn_shortcut = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '1', trainable=trainable_bn)

        self.add4 = layers.Add()
        self.relu4 = layers.Activation('relu')
        self.last = None

    def call(self, inputs, training):
        outputs = self.conv1(inputs)
        outputs = self.bn1(outputs, training=training)
        outputs = self.relu1(outputs)

        outputs = self.conv2(outputs)
        outputs = self.bn2(outputs, training=training)
        outputs = self.relu2(outputs)

        outputs = self.conv3(outputs)
        outputs = self.bn3(outputs, training=training)

        shortcut = self.conv_shortcut(inputs)
        shortcut = self.bn_shortcut(shortcut, training=training)

        outputs = self.add4([outputs, shortcut])
        self.last = outputs = self.relu4(outputs)
        return outputs


class ResNet50(tf.keras.Model):
    def __init__(self, data_format, initial_strides=2, pool_size=2, trainable_bn=True, l2=0.001):
        super(ResNet50, self).__init__(self)

        if data_format == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1

        def get_l2():
            if l2 is not None and l2 > 1e-6:
                return tf.keras.regularizers.l2(l2)
            return None

        self.pad1 = layers.ZeroPadding2D(padding=(1, 1), name='conv1_pad')
        self.cv1 = layers.Conv2D(64, (7, 7), strides=(initial_strides, initial_strides), padding='valid', kernel_initializer='he_normal', kernel_regularizer=get_l2(), name='conv1')
        self.bn1 = layers.BatchNormalization(axis=bn_axis, name='bn_conv1', trainable=trainable_bn)
        self.rl1 = layers.Activation('relu')

        self.pool_size = pool_size
        if self.pool_size is not None:
            self.pad2 = layers.ZeroPadding2D(padding=(1, 1), name='pool1_pad')
            self.mpl = layers.MaxPooling2D((3, 3), strides=(2, 2))

        self.block1_conv1 = conv_block(data_format, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1), trainable_bn=trainable_bn, l2=l2)
        self.block1_iden1 = identity_block(data_format, 3, [64, 64, 256], stage=2, block='b', trainable_bn=trainable_bn, l2=l2)
        self.block1_iden2 = identity_block(data_format, 3, [64, 64, 256], stage=2, block='c', trainable_bn=trainable_bn, l2=l2)

        self.block2_conv1 = conv_block(data_format, 3, [128, 128, 512], stage=3, block='a', trainable_bn=trainable_bn, l2=l2)
        self.block2_iden1 = identity_block(data_format, 3, [128, 128, 512], stage=3, block='b', trainable_bn=trainable_bn, l2=l2)
        self.block2_iden2 = identity_block(data_format, 3, [128, 128, 512], stage=3, block='c', trainable_bn=trainable_bn, l2=l2)
        self.block2_iden3 = identity_block(data_format, 3, [128, 128, 512], stage=3, block='d', trainable_bn=trainable_bn, l2=l2)

        self.block3_conv1 = conv_block(data_format, 3, [256, 256, 1024], stage=4, block='a', trainable_bn=trainable_bn, l2=l2)
        self.block3_iden1 = identity_block(data_format, 3, [256, 256, 1024], stage=4, block='b', trainable_bn=trainable_bn, l2=l2)
        self.block3_iden2 = identity_block(data_format, 3, [256, 256, 1024], stage=4, block='c', trainable_bn=trainable_bn, l2=l2)
        self.block3_iden3 = identity_block(data_format, 3, [256, 256, 1024], stage=4, block='d', trainable_bn=trainable_bn, l2=l2)
        self.block3_iden4 = identity_block(data_format, 3, [256, 256, 1024], stage=4, block='e', trainable_bn=trainable_bn, l2=l2)
        self.block3_iden5 = identity_block(data_format, 3, [256, 256, 1024], stage=4, block='f', trainable_bn=trainable_bn, l2=l2)

        self.block4_conv1 = conv_block(data_format, 3, [512, 512, 2048], stage=5, block='a', trainable_bn=trainable_bn, l2=l2)
        self.block4_iden1 = identity_block(data_format, 3, [512, 512, 2048], stage=5, block='b', trainable_bn=trainable_bn, l2=l2)
        self.block4_iden2 = identity_block(data_format, 3, [512, 512, 2048], stage=5, block='c', trainable_bn=trainable_bn, l2=l2)

    def call(self, inputs, training):
        inputs = self.pad1(inputs)
        inputs = self.cv1(inputs)
        inputs = self.bn1(inputs, training=training)
        inputs = self.rl1(inputs)
        self.pre_pool = inputs

        if self.pool_size is not None:
            inputs = self.pad2(inputs)
            inputs = self.mpl(inputs)

        inputs = self.block1_conv1(inputs, training=training)
        inputs = self.block1_iden1(inputs, training=training)
        inputs = self.block1_iden2(inputs, training=training)

        inputs = self.block2_conv1(inputs, training=training)
        inputs = self.block2_iden1(inputs, training=training)
        inputs = self.block2_iden2(inputs, training=training)
        inputs = self.block2_iden3(inputs, training=training)

        inputs = self.block3_conv1(inputs, training=training)
        inputs = self.block3_iden1(inputs, training=training)
        inputs = self.block3_iden2(inputs, training=training)
        inputs = self.block3_iden3(inputs, training=training)
        inputs = self.block3_iden4(inputs, training=training)
        inputs = self.block3_iden5(inputs, training=training)

        inputs = self.block4_conv1(inputs, training=training)
        inputs = self.block4_iden1(inputs, training=training)
        inputs = self.block4_iden2(inputs, training=training)
        return inputs

    def load(self):
        weights_path = tf.keras.utils.get_file(
            'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
            'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
            cache_subdir='models')

        with h5py.File(weights_path, mode='r') as f:
            if 'layer_names' not in f.attrs and 'model_weights' in f:
                f = f['model_weights']

            hdf5_format.load_weights_from_hdf5_group_by_name(f, utils.recurse_layers(self), skip_mismatch=False)


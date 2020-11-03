import tensorflow as tf
import h5py

from tensorflow.keras import layers
from tensorflow.python.keras.saving import hdf5_format


class conv2d_fixed_padding(tf.keras.Model):
    def __init__(self, filters, kernel_size, strides, data_format, kernel_initializer, kernel_regularizer, name):
        super(conv2d_fixed_padding, self).__init__(self)
        
        pad_total = kernel_size - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        
        if data_format == 'channels_first':
            self.pad = [[0, 0], [0, 0], [pad_beg, pad_end], [pad_beg, pad_end]]
        else:
            self.pad = [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]]
        
        self.strides = strides
        self.conv = layers.Conv2D(filters, kernel_size, strides=strides, use_bias=False, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer, name=name, data_format=data_format, padding=('same' if strides==1 else 'valid'))
        
    def call(self, inputs):
        if self.strides > 1:
            inputs = tf.pad(inputs, self.pad)
        
        outputs = self.conv(inputs)
        return outputs
        
class bottleneck_block(tf.keras.Model):
    def __init__(self, has_projection, data_format, filters, strides, stage, block, trainable_bn, l2=0.001):
        super(bottleneck_block, self).__init__(self)
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

        self.has_projection = has_projection
        if has_projection:
            self.projection = conv2d_fixed_padding(filters * 4, kernel_size=1, strides=strides, data_format=data_format, kernel_initializer='he_normal', kernel_regularizer=get_l2(), name=conv_name_base + '2a')
        
        self.bn1 = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a', trainable=trainable_bn)
        self.relu1 = layers.Activation('relu')

        self.conv1 = conv2d_fixed_padding(filters, kernel_size=1, strides=1, data_format=data_format, kernel_initializer='he_normal', kernel_regularizer=get_l2(), name=conv_name_base + '2b')
        
        self.bn2 = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b', trainable=trainable_bn)
        self.relu2 = layers.Activation('relu')
        self.conv2 = conv2d_fixed_padding(filters, kernel_size=3, strides=strides, data_format=data_format, kernel_initializer='he_normal', kernel_regularizer=get_l2(), name=conv_name_base + '2c')
        
        self.bn3 = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c', trainable=trainable_bn)
        self.relu3 = layers.Activation('relu')
        self.conv3 = conv2d_fixed_padding(filters * 4, kernel_size=1, strides=1, data_format=data_format, kernel_initializer='he_normal', kernel_regularizer=get_l2(), name=conv_name_base + '2c')
    
        self.add = layers.Add()
        self.last = None

    def call(self, inputs, training):
        outputs = self.bn1(inputs, training=training)
        outputs = self.relu1(outputs)
        
        shortcut = inputs
        if self.has_projection:
            shortcut = self.projection(outputs)

        outputs = self.conv1(outputs)
        
        outputs = self.bn2(outputs, training=training)
        outputs = self.relu2(outputs)
        outputs = self.conv2(outputs)
        
        outputs = self.bn3(outputs, training=training)
        outputs = self.relu3(outputs)
        outputs = self.conv3(outputs)

        self.last = outputs = self.add([outputs, shortcut])
        return outputs

        
class identity_block(tf.keras.Model):
    def __init__(self, has_projection, data_format, filters, strides, stage, block, trainable_bn, l2=0.001):
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

        self.has_projection = has_projection
        if has_projection:
            self.projection = conv2d_fixed_padding(filters, kernel_size=1, strides=strides, data_format=data_format, kernel_initializer='he_normal', kernel_regularizer=get_l2(), name=conv_name_base + '2a')
        
        self.bn1 = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b', trainable=trainable_bn)
        self.relu1 = layers.Activation('relu')
        self.conv1 = conv2d_fixed_padding(filters, kernel_size=3, strides=strides, data_format=data_format, kernel_initializer='he_normal', kernel_regularizer=get_l2(), name=conv_name_base + '2c')
        
        self.bn2 = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c', trainable=trainable_bn)
        self.relu2 = layers.Activation('relu')
        self.conv2 = conv2d_fixed_padding(filters, kernel_size=3, strides=1, data_format=data_format, kernel_initializer='he_normal', kernel_regularizer=get_l2(), name=conv_name_base + '2c')
    
        self.add = layers.Add()
        self.last = None

    def call(self, inputs, training):
        outputs = self.bn1(inputs, training=training)
        outputs = self.relu1(outputs)
        
        shortcut = inputs
        if self.has_projection:
            shortcut = self.projection(outputs)

        outputs = self.conv1(outputs)
        
        outputs = self.bn2(outputs, training=training)
        outputs = self.relu2(outputs)
        outputs = self.conv2(outputs)
        
        self.last = outputs = self.add([outputs, shortcut])
        return outputs

class ResNet50V2(tf.keras.Model):
    def __init__(self, data_format, initial_strides=2, pool_size=3, trainable_bn=True, l2=0.001):
        super(ResNet50V2, self).__init__(self)

        if data_format == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1

        def get_l2():
            if l2 is not None and l2 > 1e-6:
                return tf.keras.regularizers.l2(l2)
            return None

        self.conv1 = conv2d_fixed_padding(64, kernel_size=7, strides=initial_strides, data_format=data_format, kernel_initializer='he_normal', kernel_regularizer=get_l2(), name='conv1')

        self.pool_size = pool_size
        if self.pool_size is not None:
            self.mpl = layers.MaxPooling2D(self.pool_size, strides=(2, 2), padding='SAME', data_format=data_format)

        self.block1_iden0 = bottleneck_block(has_projection=True, data_format=data_format, filters=64, strides=1, stage=2, block='a', trainable_bn=trainable_bn, l2=l2)
        self.block1_iden1 = bottleneck_block(has_projection=False, data_format=data_format, filters=64, strides=1, stage=2, block='b', trainable_bn=trainable_bn, l2=l2)
        self.block1_iden2 = bottleneck_block(has_projection=False, data_format=data_format, filters=64, strides=1, stage=2, block='c', trainable_bn=trainable_bn, l2=l2)

        self.block2_iden0 = bottleneck_block(has_projection=True, data_format=data_format, filters=128, strides=2, stage=3, block='a', trainable_bn=trainable_bn, l2=l2)
        self.block2_iden1 = bottleneck_block(has_projection=False, data_format=data_format, filters=128, strides=1, stage=3, block='b', trainable_bn=trainable_bn, l2=l2)
        self.block2_iden2 = bottleneck_block(has_projection=False, data_format=data_format, filters=128, strides=1, stage=3, block='c', trainable_bn=trainable_bn, l2=l2)
        self.block2_iden3 = bottleneck_block(has_projection=False, data_format=data_format, filters=128, strides=1, stage=3, block='d', trainable_bn=trainable_bn, l2=l2)

        self.block3_iden0 = bottleneck_block(has_projection=True, data_format=data_format, filters=256, strides=2, stage=4, block='a', trainable_bn=trainable_bn, l2=l2)
        self.block3_iden1 = bottleneck_block(has_projection=False, data_format=data_format, filters=256, strides=1, stage=4, block='b', trainable_bn=trainable_bn, l2=l2)
        self.block3_iden2 = bottleneck_block(has_projection=False, data_format=data_format, filters=256, strides=1, stage=4, block='c', trainable_bn=trainable_bn, l2=l2)
        self.block3_iden3 = bottleneck_block(has_projection=False, data_format=data_format, filters=256, strides=1, stage=4, block='d', trainable_bn=trainable_bn, l2=l2)
        self.block3_iden4 = bottleneck_block(has_projection=False, data_format=data_format, filters=256, strides=1, stage=4, block='e', trainable_bn=trainable_bn, l2=l2)
        self.block3_iden5 = bottleneck_block(has_projection=False, data_format=data_format, filters=256, strides=1, stage=4, block='f', trainable_bn=trainable_bn, l2=l2)

        self.block4_iden0 = bottleneck_block(has_projection=True, data_format=data_format, filters=512, strides=2, stage=5, block='a', trainable_bn=trainable_bn, l2=l2)
        self.block4_iden1 = bottleneck_block(has_projection=False, data_format=data_format, filters=512, strides=1, stage=5, block='b', trainable_bn=trainable_bn, l2=l2)
        self.block4_iden2 = bottleneck_block(has_projection=False, data_format=data_format, filters=512, strides=1, stage=5, block='c', trainable_bn=trainable_bn, l2=l2)
        
        self.bn1 = layers.BatchNormalization(axis=bn_axis, name='bn_conv1', trainable=trainable_bn)
        self.rl1 = layers.Activation('relu')

    def call(self, inputs, training):
        inputs = self.conv1(inputs)
        
        if self.pool_size is not None:
            inputs = self.mpl(inputs)

        inputs = self.block1_iden0(inputs, training=training)
        inputs = self.block1_iden1(inputs, training=training)
        inputs = self.block1_iden2(inputs, training=training)

        inputs = self.block2_iden0(inputs, training=training)
        inputs = self.block2_iden1(inputs, training=training)
        inputs = self.block2_iden2(inputs, training=training)
        inputs = self.block2_iden3(inputs, training=training)

        inputs = self.block3_iden0(inputs, training=training)
        inputs = self.block3_iden1(inputs, training=training)
        inputs = self.block3_iden2(inputs, training=training)
        inputs = self.block3_iden3(inputs, training=training)
        inputs = self.block3_iden4(inputs, training=training)
        inputs = self.block3_iden5(inputs, training=training)

        inputs = self.block4_iden0(inputs, training=training)
        inputs = self.block4_iden1(inputs, training=training)
        inputs = self.block4_iden2(inputs, training=training)
        
        inputs = self.bn1(inputs)
        inputs = self.rl1(inputs)
        return inputs

    def load(self, filepath):
        reader = tf.compat.v1.train.load_checkpoint(filepath)
        
        # Start restoring
        self.conv1.set_weights([reader.get_tensor('resnet_model/conv2d/kernel')])

        conv_counter = 1
        bn_counter = 0
        n_subblocks = [3, 4, 6, 3]
        for main in range(4):
            for sub in range(n_subblocks[main]):
                block = getattr(self, 'block{}_iden{}'.format(main + 1, sub))

                # BN1
                block.bn1.set_weights([
                    reader.get_tensor('resnet_model/batch_normalization{}/gamma'.format('' if bn_counter == 0 else '_{}'.format(bn_counter))),
                    reader.get_tensor('resnet_model/batch_normalization{}/beta'.format('' if bn_counter == 0 else '_{}'.format(bn_counter))),
                    reader.get_tensor('resnet_model/batch_normalization{}/moving_mean'.format('' if bn_counter == 0 else '_{}'.format(bn_counter))),
                    reader.get_tensor('resnet_model/batch_normalization{}/moving_variance'.format('' if bn_counter == 0 else '_{}'.format(bn_counter)))
                ])
                bn_counter += 1

                if sub == 0:
                    block.projection.set_weights([reader.get_tensor('resnet_model/conv2d_{}/kernel'.format(conv_counter))])
                    conv_counter += 1

                block.conv1.set_weights([reader.get_tensor('resnet_model/conv2d_{}/kernel'.format(conv_counter))])
                conv_counter += 1

                block.bn2.set_weights([
                    reader.get_tensor('resnet_model/batch_normalization_{}/gamma'.format(bn_counter)),
                    reader.get_tensor('resnet_model/batch_normalization_{}/beta'.format(bn_counter)),
                    reader.get_tensor('resnet_model/batch_normalization_{}/moving_mean'.format(bn_counter)),
                    reader.get_tensor('resnet_model/batch_normalization_{}/moving_variance'.format(bn_counter))
                ])
                bn_counter += 1

                block.conv2.set_weights([reader.get_tensor('resnet_model/conv2d_{}/kernel'.format(conv_counter))])
                conv_counter += 1

                block.bn3.set_weights([
                    reader.get_tensor('resnet_model/batch_normalization_{}/gamma'.format(bn_counter)),
                    reader.get_tensor('resnet_model/batch_normalization_{}/beta'.format(bn_counter)),
                    reader.get_tensor('resnet_model/batch_normalization_{}/moving_mean'.format(bn_counter)),
                    reader.get_tensor('resnet_model/batch_normalization_{}/moving_variance'.format(bn_counter))
                ])
                bn_counter += 1

                block.conv3.set_weights([reader.get_tensor('resnet_model/conv2d_{}/kernel'.format(conv_counter))])
                conv_counter += 1

        self.bn1.set_weights([
            reader.get_tensor('resnet_model/batch_normalization_{}/gamma'.format(bn_counter)),
            reader.get_tensor('resnet_model/batch_normalization_{}/beta'.format(bn_counter)),
            reader.get_tensor('resnet_model/batch_normalization_{}/moving_mean'.format(bn_counter)),
            reader.get_tensor('resnet_model/batch_normalization_{}/moving_variance'.format(bn_counter))
        ])
        bn_counter += 1

        print('Restored {} convolutions and {} batch norms'.format(bn_counter, conv_counter))
    

class ResNet18V2(tf.keras.Model):
    def __init__(self, data_format, initial_strides=2, pool_size=3, trainable_bn=True, l2=0.001):
        super(ResNet18V2, self).__init__(self)

        if data_format == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1

        def get_l2():
            if l2 is not None and l2 > 1e-6:
                return tf.keras.regularizers.l2(l2)
            return None

        self.conv1 = conv2d_fixed_padding(64, kernel_size=7, strides=initial_strides, data_format=data_format, kernel_initializer='he_normal', kernel_regularizer=get_l2(), name='conv1')

        self.pool_size = pool_size
        if self.pool_size is not None:
            self.mpl = layers.MaxPooling2D(self.pool_size, strides=(2, 2), padding='SAME', data_format=data_format)

        self.block1_iden0 = identity_block(has_projection=True, data_format=data_format, filters=64, strides=1, stage=2, block='a', trainable_bn=trainable_bn, l2=l2)
        self.block1_iden1 = identity_block(has_projection=False, data_format=data_format, filters=64, strides=1, stage=2, block='b', trainable_bn=trainable_bn, l2=l2)

        self.block2_iden0 = identity_block(has_projection=True, data_format=data_format, filters=128, strides=2, stage=3, block='a', trainable_bn=trainable_bn, l2=l2)
        self.block2_iden1 = identity_block(has_projection=False, data_format=data_format, filters=128, strides=1, stage=3, block='b', trainable_bn=trainable_bn, l2=l2)

        self.block3_iden0 = identity_block(has_projection=True, data_format=data_format, filters=256, strides=2, stage=4, block='a', trainable_bn=trainable_bn, l2=l2)
        self.block3_iden1 = identity_block(has_projection=False, data_format=data_format, filters=256, strides=1, stage=4, block='b', trainable_bn=trainable_bn, l2=l2)

        self.block4_iden0 = identity_block(has_projection=True, data_format=data_format, filters=512, strides=2, stage=5, block='a', trainable_bn=trainable_bn, l2=l2)
        self.block4_iden1 = identity_block(has_projection=False, data_format=data_format, filters=512, strides=1, stage=5, block='b', trainable_bn=trainable_bn, l2=l2)
        
        self.bn1 = layers.BatchNormalization(axis=bn_axis, name='bn_conv1', trainable=trainable_bn)
        self.rl1 = layers.Activation('relu')

    def call(self, inputs, training):
        inputs = self.conv1(inputs)
        
        if self.pool_size is not None:
            inputs = self.mpl(inputs)

        inputs = self.block1_iden0(inputs, training=training)
        inputs = self.block1_iden1(inputs, training=training)

        inputs = self.block2_iden0(inputs, training=training)
        inputs = self.block2_iden1(inputs, training=training)

        inputs = self.block3_iden0(inputs, training=training)
        inputs = self.block3_iden1(inputs, training=training)

        inputs = self.block4_iden0(inputs, training=training)
        inputs = self.block4_iden1(inputs, training=training)
        
        inputs = self.bn1(inputs)
        inputs = self.rl1(inputs)
        return inputs

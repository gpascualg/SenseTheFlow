import tensorflow as tf


class GroupNorm(tf.layers.Layer):
  def __init__(self, G, esp, data_format='channels_last', **kwargs):
    self.G = G
    self.esp = esp
    self.data_format = data_format
    
    super(GroupNorm, self).__init__(**kwargs)

  def build(self, input_shape):
    # Create a trainable weight variable for this layer.
    self.gamma = self.add_variable(
        name='gamma', 
        shape=(input_shape[1 if self.data_format == 'channels_first' else 3],),
        initializer=tf.constant_initializer(1.0),
        trainable=True
    )
    self.beta = self.add_variable(
        name='beta', 
        shape=(input_shape[1 if self.data_format == 'channels_first' else 3],),
        initializer=tf.constant_initializer(0.0),
        trainable=True
    )
    
    super(GroupNorm, self).build(input_shape)  # Be sure to call this somewhere!

  def call(self, x):
    if self.data_format != 'channels_first':
      x = tf.transpose(x, [0, 3, 1, 2])
      
    N, C, H, W = x.shape.as_list()
    if N is None:
      N = tf.shape(x)[0]

    G = min(self.G, C)
    x = tf.reshape(x, [N, G, C // G, H, W])
    mean, var = tf.nn.moments(x, [2, 3, 4], keep_dims=True)
    x = (x - mean) / tf.sqrt(var + self.esp)
    
    gamma = tf.reshape(self.gamma, [1, C, 1, 1])
    beta = tf.reshape(self.beta, [1, C, 1, 1])

    output = tf.reshape(x, [N, C, H, W]) * gamma + beta
        
    # tranpose: [bs, c, h, w] to [bs, h, w, c] following the paper
    if self.data_format != 'channels_first':
      output = tf.transpose(output, [0, 2, 3, 1])
    
    return output

  def compute_output_shape(self, input_shape):
    return input_shape

def norm(x, norm_type, is_training, data_format, G=32, esp=1e-5):
  if norm_type == 'none':
    output = x
  elif norm_type == 'batch':
    output = tf.layers.batch_norm(
      x, center=True, scale=True, decay=0.999,
      is_training=is_training, updates_collections=None
    )
  elif norm_type == 'group':
    output = GroupNorm(G, esp, data_format)(x)
  else:
    raise NotImplementedError
   
  return output

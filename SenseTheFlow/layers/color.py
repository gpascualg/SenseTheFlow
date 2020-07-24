import matplotlib
import matplotlib.cm
import tensorflow as tf
import numpy as np

def _colorize(value, vmin=None, vmax=None, norm=False, cmap=None):
    """
    A utility function for TensorFlow that maps a grayscale image to a matplotlib
    colormap for use with TensorBoard image summaries.
    By default it will normalize the input value to the range 0..1 before mapping
    to a grayscale colormap.
    Arguments:
      - value: 2D Tensor of shape [height, width] or 3D Tensor of shape
        [height, width, 1].
      - vmin: the minimum value of the range used for normalization.
        (Default: value minimum)
      - vmax: the maximum value of the range used for normalization.
        (Default: value maximum)
      - cmap: a valid cmap named for use with matplotlib's `get_cmap`.
        (Default: 'gray')
    Example usage:
    ```
    output = tf.random_uniform(shape=[256, 256, 1])
    output_color = colorize(output, vmin=0.0, vmax=1.0, cmap='viridis')
    tf.summary.image('output', output_color)
    ```
    
    Returns a 3D tensor of shape [height, width, 3].
    """
    
    # normalize
    if norm:
        vmin = tf.reduce_min(value) if vmin is None else vmin
        vmax = tf.reduce_max(value) if vmax is None else vmax
        value = (value - vmin) / (vmax - vmin) # vmin..vmax
    else:
        value = tf.clip_by_value(value, vmin, vmax) / vmax

    # squeeze last dim if it exists
    value = tf.squeeze(value)

    # quantize
    indices = tf.to_int32(tf.round(value * 255))

    # gather
    cm = matplotlib.cm.get_cmap(cmap if cmap is not None else 'gray')
    colors = cm(np.arange(256))[:, :3]
    colors = tf.constant(colors, dtype=tf.float32)
    value = tf.gather(colors, indices)
    value = tf.expand_dims(value, axis=0)
    
    return value

def colorize(value, vmin=None, vmax=None, norm=False, cmap=None):
    return tf.cond(
        tf.equal(tf.rank(value), 4),
        lambda: _colorize(tf.expand_dims(value[0, :, :, :], axis=0), vmin=vmin, vmax=vmax, norm=norm, cmap=cmap),
        lambda: _colorize(value,                                     vmin=vmin, vmax=vmax, norm=norm, cmap=cmap)
    )

def hex2rgb(hexcode):
    rgb = tuple([int(hexcode[1+i*2:3+i*2], 16)/255.0 for i in range(3)])
    return rgb

def norm(l):
    return tuple([e / 255.0 for e in l])

def _decode(enc, class_colors_tuple=None, class_colors_hex=None):
    assert class_colors_tuple is not None or class_colors_hex is not None, "Either class_colors_tuple or class_colors_hex must be specified"
    
    if class_colors_tuple:
        class_colors = [norm(c) for c in class_colors_tuple]
    else:
        class_colors = [hex2rgb(c) for c in class_colors_hex]
    
    num_cls = len(class_colors)
    enc = tf.cast(tf.squeeze(enc), tf.int32)
    onehot = tf.expand_dims(tf.one_hot(enc, num_cls), axis=2)
    colors = np.asarray([class_colors[cls] for cls in range(num_cls)]).reshape((1, 1, num_cls, 3))
    colors = np.swapaxes(colors, 2, 3)
    colors = tf.constant(colors, dtype=tf.float32, name='colors')
    final = onehot * colors
    outputs = tf.expand_dims(tf.reduce_sum(final, axis=-1), axis=0)
    return outputs

def decode(enc, class_colors_tuple=None, class_colors_hex=None):
    return tf.cond(
        tf.equal(tf.rank(enc), 4),
        lambda: _decode(tf.expand_dims(enc[0, :, :, :], axis=0), class_colors_tuple=class_colors_tuple, class_colors_hex=class_colors_hex),
        lambda: _decode(enc,                                     class_colors_tuple=class_colors_tuple, class_colors_hex=class_colors_hex)
    )

def decode_np(enc, class_colors_tuple=None, class_colors_hex=None):
    assert class_colors_tuple is not None or class_colors_hex is not None, "Either class_colors_tuple or class_colors_hex must be specified"
    
    if class_colors_tuple:
        class_colors = [norm(c) for c in class_colors_tuple]
    else:
        class_colors = [hex2rgb(c) for c in class_colors_hex]
        
    num_cls = len(class_colors)
    
    return np.take(class_colors, enc, axis=0)

# @ https://github.com/affinelayer/pix2pix-tensorflow/blob/master/pix2pix.py
def check_image(image):
    assertion = tf.assert_equal(tf.shape(image)[-1], 3, message="image must have 3 color channels")
    with tf.control_dependencies([assertion]):
        image = tf.identity(image)

    if image.get_shape().ndims not in (3, 4):
        raise ValueError("image must be either 3 or 4 dimensions")

    # make the last dimension 3 so that you can unstack the colors
    shape = list(image.get_shape())
    shape[-1] = 3
    image.set_shape(shape)
    return image

def rgb_to_lab(srgb):
    with tf.name_scope("rgb_to_lab"):
        srgb = check_image(srgb)
        srgb_pixels = tf.reshape(srgb, [-1, 3])

        with tf.name_scope("srgb_to_xyz"):
            linear_mask = tf.cast(srgb_pixels <= 0.04045, dtype=tf.float32)
            exponential_mask = tf.cast(srgb_pixels > 0.04045, dtype=tf.float32)
            rgb_pixels = (srgb_pixels / 12.92 * linear_mask) + (((srgb_pixels + 0.055) / 1.055) ** 2.4) * exponential_mask
            rgb_to_xyz = tf.constant([
                #    X        Y          Z
                [0.412453, 0.212671, 0.019334], # R
                [0.357580, 0.715160, 0.119193], # G
                [0.180423, 0.072169, 0.950227], # B
            ])
            xyz_pixels = tf.matmul(rgb_pixels, rgb_to_xyz)

        # https://en.wikipedia.org/wiki/Lab_color_space#CIELAB-CIEXYZ_conversions
        with tf.name_scope("xyz_to_cielab"):
            # convert to fx = f(X/Xn), fy = f(Y/Yn), fz = f(Z/Zn)

            # normalize for D65 white point
            xyz_normalized_pixels = tf.multiply(xyz_pixels, [1/0.950456, 1.0, 1/1.088754])

            epsilon = 6/29
            linear_mask = tf.cast(xyz_normalized_pixels <= (epsilon**3), dtype=tf.float32)
            exponential_mask = tf.cast(xyz_normalized_pixels > (epsilon**3), dtype=tf.float32)
            fxfyfz_pixels = (xyz_normalized_pixels / (3 * epsilon**2) + 4/29) * linear_mask + (xyz_normalized_pixels ** (1/3)) * exponential_mask

            # convert to lab
            fxfyfz_to_lab = tf.constant([
                #  l       a       b
                [  0.0,  500.0,    0.0], # fx
                [116.0, -500.0,  200.0], # fy
                [  0.0,    0.0, -200.0], # fz
            ])
            lab_pixels = tf.matmul(fxfyfz_pixels, fxfyfz_to_lab) + tf.constant([-16.0, 0.0, 0.0])

        return tf.reshape(lab_pixels, tf.shape(srgb))


def lab_to_rgb(lab):
    with tf.name_scope("lab_to_rgb"):
        lab = check_image(lab)
        lab_pixels = tf.reshape(lab, [-1, 3])

        # https://en.wikipedia.org/wiki/Lab_color_space#CIELAB-CIEXYZ_conversions
        with tf.name_scope("cielab_to_xyz"):
            # convert to fxfyfz
            lab_to_fxfyfz = tf.constant([
                #   fx      fy        fz
                [1/116.0, 1/116.0,  1/116.0], # l
                [1/500.0,     0.0,      0.0], # a
                [    0.0,     0.0, -1/200.0], # b
            ])
            fxfyfz_pixels = tf.matmul(lab_pixels + tf.constant([16.0, 0.0, 0.0]), lab_to_fxfyfz)

            # convert to xyz
            epsilon = 6/29
            linear_mask = tf.cast(fxfyfz_pixels <= epsilon, dtype=tf.float32)
            exponential_mask = tf.cast(fxfyfz_pixels > epsilon, dtype=tf.float32)
            xyz_pixels = (3 * epsilon**2 * (fxfyfz_pixels - 4/29)) * linear_mask + (fxfyfz_pixels ** 3) * exponential_mask

            # denormalize for D65 white point
            xyz_pixels = tf.multiply(xyz_pixels, [0.950456, 1.0, 1.088754])

        with tf.name_scope("xyz_to_srgb"):
            xyz_to_rgb = tf.constant([
                #     r           g          b
                [ 3.2404542, -0.9692660,  0.0556434], # x
                [-1.5371385,  1.8760108, -0.2040259], # y
                [-0.4985314,  0.0415560,  1.0572252], # z
            ])
            rgb_pixels = tf.matmul(xyz_pixels, xyz_to_rgb)
            # avoid a slightly negative number messing up the conversion
            rgb_pixels = tf.clip_by_value(rgb_pixels, 0.0, 1.0)
            linear_mask = tf.cast(rgb_pixels <= 0.0031308, dtype=tf.float32)
            exponential_mask = tf.cast(rgb_pixels > 0.0031308, dtype=tf.float32)
            srgb_pixels = (rgb_pixels * 12.92 * linear_mask) + ((rgb_pixels ** (1/2.4) * 1.055) - 0.055) * exponential_mask

        return tf.reshape(srgb_pixels, tf.shape(lab))

import tensorflow as tf


def squared_op(vec, ndims, op=tf.math.add):
    ones = tf.matmul(tf.ones_like(vec), tf.transpose(tf.ones_like(vec))) / ndims
    ones = tf.expand_dims(ones, axis=-1)
    ones = tf.concat([ones] * ndims, axis=-1)
    vec = tf.expand_dims(vec, axis=1)
    vec_squared = vec * ones
    
    return op(vec_squared, tf.transpose(vec_squared, perm=[1, 0, 2]))

def diagonalize(sigma, ndims):
    eye = tf.expand_dims(tf.eye(ndims), axis=-1)
    sigma = tf.expand_dims(tf.transpose(sigma), axis=1)
    return eye * sigma

def squared_sigma(diag, op=tf.math.add):
    ones = tf.expand_dims(tf.ones_like(diag), axis=-1)
    ones = ones * tf.transpose(ones, perm=[0, 1, 3, 2])
    
    ext = ones * tf.expand_dims(diag, axis=-1)
    summed = op(ext, tf.transpose(ext, perm=[0, 1, 3, 2]))
    summed = tf.transpose(summed, perm=[2, 3, 0, 1])
    return summed

def diag_inverse(diag):
    mask = tf.where(tf.greater(diag, 0.0), tf.ones_like(diag), tf.zeros_like(diag))
    return 1.0 / (diag + 1.0 - mask) * mask

def det_diag(diag, keepdims=False):
    diag = tf.linalg.diag_part(diag)
    return tf.reduce_prod(diag, axis=-1, keepdims=keepdims)

def pairwise_bhattacharyya(dist, diagonal=True):
    mu = dist.mean()
    ndim = int(dist.batch_shape[1])
        
    smu = squared_op(mu, ndim, op=tf.subtract)
    if diagonal:
        diag = diagonalize(dist.stddev(), ndim)
    else:
        diag = dist.covariance()
        diag = tf.transpose(diag, perm=[1, 2, 0])
            
    diagsq = squared_sigma(diag, op=lambda x, y: tf.math.add(x, y) / 2.0)
    diaginv = diag_inverse(diagsq)

    # Compute first term
    smu_v = tf.expand_dims(smu, axis=2)
    smu_h = tf.expand_dims(smu, axis=-1)
    dist = tf.matmul(tf.matmul(smu_v, diaginv), smu_h)
        
    term_a = 1 / 8.0 * dist
    term_a = tf.squeeze(tf.squeeze(term_a, axis=-1), axis=-1)
    
    det_all = det_diag(diagsq)
    det_single = det_diag(tf.transpose(diag, perm=[2, 0, 1]), keepdims=True)
    det_sq = tf.matmul(det_single, tf.transpose(det_single))
    
    #term_b = 1 / 2.0 * tf.log(tf.maximum(1e-6, det_all / tf.sqrt(det_sq)))
    frac_det = tf.sqrt(det_sq)
    frac_det_mask = tf.where(tf.greater(frac_det, 0.0), tf.ones_like(frac_det), tf.zeros_like(frac_det))
    
    frac = det_all / (frac_det + (1.0 - frac_det_mask)) * frac_det_mask
    frac_mask = tf.where(tf.greater(frac, 0.0), tf.zeros_like(frac), tf.ones_like(frac))
    
    log = tf.log(frac + frac_mask)
    term_b = 1 / 2.0 * log
    
    return term_a + term_b

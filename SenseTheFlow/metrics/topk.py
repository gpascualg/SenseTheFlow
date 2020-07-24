import tensorflow as tf

def topk(labels, distances, num_classes, k):
    _, topk_indices = tf.nn.top_k(-distances, k=k)
    topk_indices = tf.gather(labels, topk_indices)
    mask_labels = tf.one_hot(labels, num_classes)
    mask_labels = tf.expand_dims(mask_labels, axis=1)
    mask_labels = tf.concat([mask_labels]*k, axis=1)

    mask_predictions = tf.one_hot(topk_indices, num_classes)

    common_elements = tf.multiply(mask_predictions, mask_labels)
    correct_indices = tf.reduce_max(common_elements, axis=[1, 2])
    num_matches = tf.reduce_mean(correct_indices)
    return num_matches

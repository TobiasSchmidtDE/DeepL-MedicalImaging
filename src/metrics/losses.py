import tensorflow as tf


def weighted_bce(y_true, y_pred):
    """input: column with one hot encoded labels"""
    # we can later add a dict with weight constants for each dataset
    p_weight = 0.5
    n_weight = 0.5
    squared_difference = -(p_weight * y_true * tf.math.log(y_pred) +
                           n_weight * (1 - y_true) * (1 - tf.math.log(y_pred)))
    return tf.reduce_mean(squared_difference, axis=-1)

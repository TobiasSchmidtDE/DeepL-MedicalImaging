import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import Loss
from tensorflow.python.ops.losses import util as tf_losses_util
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.keras import backend_config
from tensorflow.python.keras import backend as K


class WeightedBinaryCrossentropy(Loss):
    def __init__(self, positive_class_weights, negative_class_weights):
        super().__init__(
            name="weighted_binary_crossentropy")

        assert positive_class_weights.shape == negative_class_weights.shape, \
            "positive and negative class weights must have the same shape"

        self.positive_class_weights = positive_class_weights
        self.negative_class_weights = negative_class_weights
        self.epsilon = backend_config.epsilon

    def call(self, y_true, y_pred):
        assert self.positive_class_weights.dtype == self.negative_class_weights.dtype, \
            "positive and negative class weights must have the same dtype"
        assert y_pred.shape[-1] == len(self.positive_class_weights), \
            "Number of classes in prediction doesn't match number of class_weights"
        assert y_pred.dtype == self.positive_class_weights.dtype, \
            "y_pred and class weights must have the same dtype"

        if tensor_util.is_tensor(y_pred) and tensor_util.is_tensor(y_true):
            y_pred, y_true = tf_losses_util.squeeze_or_expand_dimensions(
                y_pred, y_true)

        epsilon_ = constant_op.constant(
            self.epsilon(), dtype=y_pred.dtype.base_dtype)
        y_pred = clip_ops.clip_by_value(y_pred, epsilon_, 1. - epsilon_)

        # Create a mask for which labels are provided and which are not (e.g. NaN encoded as -1)
        # where 0 means no label (-1) and 1 means a label was provided (0 or 1)
        mask = tf.cast(tf.math.greater_equal(
            y_true, 0), y_true.dtype.base_dtype)

        # Compute cross entropy from probabilities.
        bce_pos = y_true * math_ops.log(y_pred + epsilon_)
        bce_neg = (1 - y_true) * math_ops.log(1 - y_pred + epsilon_)

        # removes all nans of bce_pos and bce_neg
        bce_pos = tf.math.multiply_no_nan(bce_pos, mask)
        bce_neg = tf.math.multiply_no_nan(bce_neg, mask)

        bce = bce_pos * self.positive_class_weights + \
            bce_neg * self.negative_class_weights

        # caclulate mean, but only weighted by the number of classes
        return tf.reduce_sum(-bce) / tf.reduce_sum(mask)
        # return K.mean(-bce)


class BinaryCrossentropy(Loss):
    def __init__(self):
        super().__init__(
            name="custom_binary_crossentropy")

        self.epsilon = lambda: 1e-5
        print(f"Initialzed {self.name} with epsilon {self.epsilon()}")

    def call(self, y_true, y_pred):

        epsilon_ = constant_op.constant(
            self.epsilon(), dtype=y_pred.dtype.base_dtype)
        #y_pred = clip_ops.clip_by_value(y_pred, epsilon_, 1. - epsilon_)
        if not tf.reduce_all((y_pred <= 1)):
            print("Some predictions are greater than 1")
        if not tf.reduce_all((y_pred >= 0)):
            print("Some predictions are less than 0")

        # Compute cross entropy from probabilities.
        bce = y_true * math_ops.log(y_pred + epsilon_)
        bce += (1 - y_true) * math_ops.log(1 - y_pred + epsilon_)

        return K.mean(-bce)


def compute_class_weight(datagenerator):
    """
    Calculates the class weightes for each class based on the number
    of positive occurances (e.g. 5 examples contained the class 'c' as label)
    and negative occurances (e.g. 10 examples did not contain the class 'c' as label)
    of the respective class.

    Parameters:
        datagenerator (ImageDataGenerator):
            The datagenerator to provide the samples and their labels (distribution)
            from which the class labels should be dervied from

    Returns:
        class_weights_positive (tf.tensor(shape=(num_classes,))):
            The weights based on the positive occurances of the class labels
        class_weights_negative (tf.tensor(shape=(num_classes,))):
            The weights based on the negative occurances of the class labels
    """
    labels = np.copy(datagenerator.get_labels())

    # set -1 to nan so it is disregarded by the coming compuations
    labels[labels == -1] = np.nan

    num_samples, num_classes = labels.shape
    class_weights = [0, ]*num_classes
    class_weights_positive = [0, ]*num_classes
    class_weights_negative = [0, ]*num_classes

    # we encode all classes with their position +1
    # that way 0s are negatives and any positive number is the positive occurance of this class
    sparse_positive_labels = labels * (np.array(range(num_classes)) + 1)

    # here we encode all negative occurances with their class id
    sparse_negativ_labels = (1-labels) * (np.array(range(num_classes)) + 1)
    for i in range(num_classes):
        class_id = i + 1
        num_positive_occurence = len(
            sparse_positive_labels[sparse_positive_labels == class_id])
        num_negative_occurence = len(
            sparse_negativ_labels[sparse_negativ_labels == class_id])

        class_weights[i] = num_samples / num_positive_occurence
        class_weights_positive[i] = (
            num_positive_occurence+num_negative_occurence) / num_positive_occurence
        class_weights_negative[i] = (
            num_positive_occurence+num_negative_occurence) / num_negative_occurence
    #class_weights = class_weights / (np.array(class_weights).mean())
    scale_factor = 10
    return tf.keras.utils.normalize(tf.constant(class_weights, dtype=tf.float32), order=1)[0] * scale_factor,\
        tf.keras.utils.normalize(tf.constant(class_weights_positive, dtype=tf.float32), order=1)[0] * scale_factor,\
        tf.keras.utils.normalize(tf.constant(
            class_weights_negative, dtype=tf.float32), order=1)[0] * scale_factor,

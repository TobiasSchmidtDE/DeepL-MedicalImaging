import numpy as np
import tensorflow as tf
import keras.backend.tensorflow_backend as K
from tensorflow.keras.losses import Loss
from tensorflow.python.ops.losses import util as tf_losses_util
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.keras import backend_config

class SimpleWeightedBinaryCrossentropy(Loss):
    def __init__(self,class_weights):
        super(SimpleWeightedBinaryCrossentropy, self).__init__(name="simple_weighted_binary_crossentropy")
        self.class_weights = class_weights
        
    def call(self, y_true, y_pred):
        if tensor_util.is_tensor(y_pred) and tensor_util.is_tensor(y_true):
              y_pred, y_true = tf_losses_util.squeeze_or_expand_dimensions(y_pred, y_true)
        bce = K.binary_crossentropy(y_true, y_pred)
        
        assert bce.shape[-1] == len(self.class_weights), "Number of classes in prediction doesn't match number of class_weights"
        return K.mean(bce*self.class_weights)
    

class WeightedBinaryCrossentropy(Loss):
    def __init__(self, positive_class_weights, negative_class_weights):
        super(WeightedBinaryCrossentropy, self).__init__(name="weighted_binary_crossentropy")
        
        assert positive_class_weights.shape == negative_class_weights.shape, "positive and negative class weights must have the same shape"

        self.positive_class_weights = positive_class_weights
        self.negative_class_weights = negative_class_weights
        self.epsilon = backend_config.epsilon
        
    def call(self, y_true, y_pred):
        assert self.positive_class_weights.dtype == self.negative_class_weights.dtype,"positive and negative class weights must have the same dtype"
        assert y_pred.shape[-1] == len(self.positive_class_weights), "Number of classes in prediction doesn't match number of class_weights"
        assert y_pred.dtype == self.positive_class_weights.dtype, "y_pred and class weights must have the same dtype"
        
        if tensor_util.is_tensor(y_pred) and tensor_util.is_tensor(y_true):
              y_pred, y_true = tf_losses_util.squeeze_or_expand_dimensions(y_pred, y_true)
        
        epsilon_ = constant_op.constant(self.epsilon(), dtype=y_pred.dtype.base_dtype)
        y_pred = clip_ops.clip_by_value(y_pred, epsilon_, 1. - epsilon_)

        # Compute cross entropy from probabilities.
        bce_pos = y_true * math_ops.log(y_pred + epsilon_)
        bce_neg = (1 - y_true) * math_ops.log(1 - y_pred + epsilon_)
        bce = bce_pos * self.positive_class_weights + bce_neg * self.negative_class_weights 
        
        return K.mean(-bce)
    
  
def compute_class_weight(datagenerator):
    labels = datagenerator.get_labels()
    num_samples, num_classes = labels.shape
    class_weights_positive = [0,]*num_classes
    class_weights_negative = [0,]*num_classes
    # we encode all classes with their position +1
    # that way 0s are negatives and any positive number is the positive occurance of this class
    sparse_positive_labels = labels * (np.array(range(num_classes)) + 1)
    # here we encode all negative occurances with their class id
    sparse_negativ_labels = (1-labels) * (np.array(range(num_classes)) + 1)
    for i in range(num_classes):
        class_id = i +1
        num_positive_occurence = len(sparse_positive_labels[sparse_positive_labels == class_id])
        num_negative_occurence = len(sparse_negativ_labels[sparse_negativ_labels == class_id])
        class_weights_positive[i] = (num_positive_occurence+num_negative_occurence) / num_positive_occurence
        class_weights_negative[i] = (num_positive_occurence+num_negative_occurence) / num_negative_occurence
    return tf.constant(class_weights_positive, dtype=tf.float32),tf.constant(class_weights_negative, dtype=tf.float32)

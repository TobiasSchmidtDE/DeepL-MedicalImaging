import tensorflow as tf


def FBetaScore(precision, recall, beta):
    return (1+beta**2) * tf.math.divide_no_nan((precision * recall), (beta**2 * precision + recall))


class F2Score(tf.keras.metrics.Metric):
    def __init__(self, thresholds=None, top_k=None, class_id=None, name=None, dtype=None):
        super(F2Score, self).__init__(name=name, dtype=dtype)
        self.precision_metric = tf.keras.metrics.Precision(thresholds=thresholds,
                                                           top_k=top_k,
                                                           class_id=class_id,
                                                           name=name,
                                                           dtype=dtype)
        self.recall_metric = tf.keras.metrics.Recall(thresholds=thresholds,
                                                     top_k=top_k,
                                                     class_id=class_id,
                                                     name=name,
                                                     dtype=dtype)

    def update_state(self, y_true, y_pred, **kwargs):  # pylint: disable=W0221
        self.precision_metric.update_state(
            y_true, y_pred, **kwargs)
        self.recall_metric.update_state(
            y_true, y_pred, **kwargs)

    def result(self):
        return FBetaScore(self.precision_metric.result(), self.recall_metric.result(), 2)

    def reset_states(self):
        self.precision_metric.reset_states()
        self.recall_metric.reset_states()


class SingleClassMetric(tf.keras.metrics.Metric):
    def __init__(self, base_metric, class_id, class_name=None, dtype=None, **vargs):
        name = base_metric.name + "_" + \
            str(class_id if class_name is None else class_name)
        super().__init__(
            name=name, dtype=dtype, **vargs)
        # initalize a fresh instance of the metric. Otherwise the same base_metric instance
        # might be shared between different single_class_metrics and interfere with each other
        self.base_metric = base_metric.__class__(name=base_metric.name)
        self.class_id = class_id

    def update_state(self, y_true, y_pred, sample_weight=None):  # pylint: disable=W0221
        # Create a mask for which labels are provided and which are not (e.g. NaN encoded as -1)
        # where 0 means no label (-1) and 1 means a label was provided (0 or 1)
        #mask = tf.cast(tf.math.greater_equal(y_true, 0), y_true.dtype.base_dtype)
        #y_true = tf.math.multiply_no_nan(y_true, mask)
        self.base_metric.update_state(
            y_true[:, self.class_id], y_pred[:, self.class_id])

    def result(self):
        return self.base_metric.result()

    def reset_states(self):
        self.base_metric.reset_states()


class NaNWrapper(tf.keras.metrics.Metric):
    def __init__(self, base_metric, *args, **vargs):
        # initalize a fresh instance of the metric. Otherwise the same base_metric instance
        # might be shared between different wrapper instances and interfere with each other
        self.base_metric = base_metric
        vargs["name"] = self.base_metric.name
        super().__init__(**vargs)

    def update_state(self, y_true, y_pred, sample_weight=None):  # pylint: disable=W0221
        # Create a mask for which labels are provided and which are not (e.g. NaN encoded as -1)
        # where 0 means no label (-1) and 1 means a label was provided (0 or 1)
        mask = tf.cast(tf.math.greater_equal(
            y_true, 0), y_true.dtype.base_dtype)
        y_true = tf.math.multiply_no_nan(y_true, mask)
        self.base_metric.update_state(y_true, y_pred)

    def result(self):
        return self.base_metric.result()

    def reset_states(self):
        self.base_metric.reset_states()

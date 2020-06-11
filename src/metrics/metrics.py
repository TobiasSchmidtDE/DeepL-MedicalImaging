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

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision_metric.update_state(
            y_true, y_pred, sample_weight=sample_weight)
        self.recall_metric.update_state(
            y_true, y_pred, sample_weight=sample_weight)

    def result(self):
        return FBetaScore(self.precision_metric.result(), self.recall_metric.result(), 2)

    def reset_states(self):
        self.precision_metric.reset_states()
        self.recall_metric.reset_states()


# WORK IN PROGRESS!
# Class not finished, throwing exception in update_state method when used in training
class SingleClassMetric(tf.keras.metrics.Metric):
    def __init__(self, base_metric, class_id, class_name=None, dtype=None, **vargs):
        name = base_metric.name + "_" + \
            str(class_id if class_name is None else class_name)
        super(SingleClassMetric, self).__init__(
            name=name, dtype=dtype, **vargs)
        # initalize a fresh instance of the metric. Otherwise the same base_metric instance
        # might be shared between different single_class_metrics and interfere with each other
        self.base_metric = base_metric.__class__(name=base_metric.name)
        self.class_id = class_id

    def update_state(self, y_true, y_pred, sample_weight=None):
        class_num = y_pred.shape[-1]
        sample_weight = tf.one_hot(self.class_id,
                                   class_num,
                                   on_value=1.0,
                                   off_value=0.0)
        self.base_metric.update_state(
            y_true, y_pred, sample_weight=sample_weight)

    def result(self):
        return self.base_metric.result()

    def reset_states(self):
        self.base_metric.reset_states()

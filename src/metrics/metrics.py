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

    def update_state(self, y_true, y_pred, sample_weight=None):  # pylint: disable=W:279
        self.precision_metric.update_state(
            y_true, y_pred, sample_weight=sample_weight)
        self.recall_metric.update_state(
            y_true, y_pred, sample_weight=sample_weight)

    def result(self):
        return FBetaScore(self.precision_metric.result(), self.recall_metric.result(), 2)

    def reset_states(self):
        self.precision_metric.reset_states()
        self.recall_metric.reset_states()

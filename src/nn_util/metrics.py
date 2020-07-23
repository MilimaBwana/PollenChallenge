import tensorflow as tf
import numpy as np


class MultiClassPrecision(tf.keras.metrics.Metric):
    """Precision for more than 2 classes.
    Arguments:
        num_classes: number of classes.
        reduce: reduction type. If 'macro', the precision is computed independently for each class and
            then averaged (hence treating all classes equally). If 'micro', the number of samples per class
            are used as weights to compute the average precision.
        top_k: if set, then so only top_k classes regarding frequency are
            considered for calculating the average precision. If top_k is negative, the bottom_k classes regarding
            frequency are considered for calculating the average precision.
        name: name of this metric.
    """

    def __init__(self, num_classes, reduce="macro", top_k=np.Inf,
                 name="multiclass_precision", **kwargs):
        super().__init__(name=name)
        self.num_classes = num_classes

        if reduce not in ("micro", "macro"):
            raise ValueError("Unknown reduction.")
        self.reduce = reduce

        top_k = tf.cast(top_k, dtype=tf.float32)
        if tf.math.is_inf(tf.constant(top_k)):
            self.top_k = tf.cast(self.num_classes, dtype=tf.int32)
        else:
            self.top_k = tf.cast(top_k, dtype=tf.int32)
        assert (tf.math.abs(self.top_k) <= num_classes), "The top-k amount must be less than or equal to num_classes"

        self.confusion_matrix = self.add_weight(
            "conf_mtx",
            shape=(self.num_classes, self.num_classes),
            initializer=tf.keras.initializers.zeros,
            dtype=tf.float32,
        )
        self.precision = self.add_weight(name='mp', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weights=None):
        """ Updates the confusion matrix and metric value. y_true and y_pred need to have same shape.
        @param y_true: groundtruth labels.
        @param y_pred: predicted labels
        @param sample_weights: not used.
        """

        y_true = tf.squeeze(y_true)
        y_pred = tf.squeeze(y_pred)
        new_conf_mtx = tf.math.confusion_matrix(
            labels=y_true,
            predictions=y_pred,
            num_classes=self.num_classes,
            dtype=tf.float32,
        )
        self.confusion_matrix.assign_add(new_conf_mtx)

    def result(self):
        """
        @return Returns the current value of precision, dependent on the reduce type.
        """
        precision_per_class = tf.math.divide_no_nan(tf.linalg.diag_part(self.confusion_matrix),
                                                    tf.math.reduce_sum(self.confusion_matrix, axis=0))
        idx_top_k = _get_top_k_cm_indices(self.confusion_matrix, self.num_classes, self.top_k)
        precision_per_class = tf.gather(precision_per_class, indices=idx_top_k, axis=0)
        if self.reduce == "macro":
            num_classes_with_samples = _get_num_classes_with_samples(self.confusion_matrix, self.num_classes,
                                                                     self.top_k)
            self.precision.assign(
                tf.math.divide_no_nan(tf.math.reduce_sum(precision_per_class),
                                      num_classes_with_samples))
        elif self.reduce == "micro":

            top_k_cm = _get_top_k_cm(self.confusion_matrix, self.num_classes, self.top_k)
            weights = tf.math.divide_no_nan(tf.reduce_sum(top_k_cm, axis=0), tf.reduce_sum(
                top_k_cm))
            self.precision.assign(tf.reduce_sum(precision_per_class * weights))

        return self.precision

    def get_confusion_matrix(self):
        return self.confusion_matrix

    def reset_states(self):
        """
        Resets confusion_matrix and precision. Should be called at the end of each epoch.
        @return Nothing
        """
        self.precision.assign(0.)
        self.confusion_matrix.assign(tf.zeros(
            shape=(self.num_classes, self.num_classes), dtype=tf.dtypes.float32, name=None
        ))


class MultiClassRecall(tf.keras.metrics.Metric):
    """ Recall for more than 2 classes.
    Arguments:
        num_classes: number of classes.
        reduce: reduction type. If 'macro', the recall is computed independently for each class and
            then averaged (hence treating all classes equally). If 'micro', the number of samples per class
            are used as weights to compute the average recall.
        top_k: if set, then so only top_k classes regarding frequency are
            considered for calculating the average recall. If top_k is negative, the bottom_k classes regarding
            frequency are considered for calculating the average recall.
        name: name of this metric.
    """

    def __init__(self, num_classes, reduce="macro", top_k=np.Inf,
                 name="multiclass_recall", **kwargs):
        super().__init__(name=name)
        self.num_classes = num_classes

        if reduce not in ("micro", "macro"):
            raise ValueError("Unknown reduction.")
        self.reduce = reduce

        top_k = tf.cast(top_k, dtype=tf.float32)
        if tf.math.is_inf(tf.constant(top_k)):
            self.top_k = tf.cast(self.num_classes, dtype=tf.int32)
        else:
            self.top_k = tf.cast(top_k, dtype=tf.int32)
        assert (tf.math.abs(self.top_k) <= num_classes), "The top-k amount must be less than or equal to num_classes"

        self.confusion_matrix = self.add_weight(
            "conf_mtx",
            shape=(self.num_classes, self.num_classes),
            initializer=tf.keras.initializers.zeros,
            dtype=tf.float32,
        )
        self.recall = self.add_weight(name='mp', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weights=None):
        """ Updates the confusion matrix and metric value. y_true and y_pred need to have same shape.
        @param y_true: groundtruth labels.
        @param y_pred: predicted labels
        @param sample_weights: not used.
        """
        y_true = tf.squeeze(y_true)
        y_pred = tf.squeeze(y_pred)
        new_conf_mtx = tf.math.confusion_matrix(
            labels=y_true,
            predictions=y_pred,
            num_classes=self.num_classes,
            dtype=tf.float32,
        )
        self.confusion_matrix.assign_add(new_conf_mtx)

    def result(self):
        """
        @return Returns the current value of recall, dependent on the reduce type.
        """
        recall_per_class = tf.math.divide_no_nan(tf.linalg.diag_part(self.confusion_matrix),
                                                 tf.math.reduce_sum(self.confusion_matrix, axis=-1))
        idx_top_k = _get_top_k_cm_indices(self.confusion_matrix, self.num_classes, self.top_k)
        recall_per_class = tf.gather(recall_per_class, indices=idx_top_k, axis=0)

        if self.reduce == "macro":
            num_classes_with_samples = _get_num_classes_with_samples(self.confusion_matrix, self.num_classes,
                                                                     self.top_k)
            self.recall.assign(
                tf.math.divide_no_nan(tf.math.reduce_sum(recall_per_class),
                                      num_classes_with_samples))
        elif self.reduce == "micro":
            """ Always yields: micro-recall=micro-precision=micro-f1score """
            top_k_cm = _get_top_k_cm(self.confusion_matrix, self.num_classes, self.top_k)
            weights = tf.math.divide_no_nan(tf.reduce_sum(top_k_cm, axis=-1), tf.reduce_sum(
                top_k_cm))
            self.recall.assign(tf.reduce_sum(recall_per_class * weights))

        return self.recall

    def get_confusion_matrix(self):
        return self.confusion_matrix

    def reset_states(self):
        """
        Resets confusion_matrix and precision. Should be called at the end of each epoch.
        @return Nothing.
        """
        self.recall.assign(0.)
        self.confusion_matrix.assign(tf.zeros(
            shape=(self.num_classes, self.num_classes), dtype=tf.dtypes.float32, name=None
        ))


class MultiClassF1Score(tf.keras.metrics.Metric):
    """ F1-Score for more than 2 classes.
    Arguments:
        num_classes: number of classes.
        reduce: reduction type. If 'macro', the f1-score is computed independently for each class and
            then averaged (hence treating all classes equally). If 'micro', the number of samples per class
            are used as weights to compute the average the f1-score.
        top_k: if set, then so only top_k classes regarding frequency are
            considered for calculating the average the f1-score. If top_k is negative, the bottom_k classes regarding
            frequency are considered for calculating the average the f1-score.
        name: name of this metric.
    """

    def __init__(self, num_classes,
                 reduce="macro", top_k=np.Inf,
                 name="multiclass_f1score", **kwargs):
        super().__init__(name=name)
        self.num_classes = num_classes

        if reduce not in ("micro", "macro"):
            raise ValueError("Unknown reduction.")
        self.reduce = reduce

        top_k = tf.cast(top_k, dtype=tf.float32)
        if tf.math.is_inf(tf.constant(top_k)):
            self.top_k = tf.cast(self.num_classes, dtype=tf.int32)
        else:
            self.top_k = tf.cast(top_k, dtype=tf.int32)
        assert (tf.math.abs(self.top_k) <= num_classes), "The top-k amount must be less than or equal to num_classes"

        self.confusion_matrix = self.add_weight(
            "conf_mtx",
            shape=(self.num_classes, self.num_classes),
            initializer=tf.keras.initializers.zeros,
            dtype=tf.float32,
        )
        self.f1score = self.add_weight(name='f1', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weights=None):
        """ Updates the confusion matrix and metric value. y_true and y_pred need to have same shape.
        @param y_true: groundtruth labels.
        @param y_pred: predicted labels
        @param sample_weights: not used.
        """
        y_true = tf.squeeze(y_true)
        y_pred = tf.squeeze(y_pred)
        new_conf_mtx = tf.math.confusion_matrix(
            labels=y_true,
            predictions=y_pred,
            num_classes=self.num_classes,
            dtype=tf.float32,
        )
        self.confusion_matrix.assign_add(new_conf_mtx)

    def get_confusion_matrix(self):
        return self.confusion_matrix

    def result(self):
        """
        @return: the current value of f1-score, dependent of reduce type.
        """
        recall_per_class = tf.math.divide_no_nan(tf.linalg.diag_part(self.confusion_matrix),
                                                 tf.math.reduce_sum(self.confusion_matrix, axis=-1))
        precision_per_class = tf.math.divide_no_nan(tf.linalg.diag_part(self.confusion_matrix),
                                                    tf.math.reduce_sum(self.confusion_matrix, axis=0))
        f1_score_per_class = tf.math.divide_no_nan((2 * tf.math.multiply(recall_per_class, precision_per_class)),
                                                   tf.math.add(
                                                       recall_per_class, precision_per_class))
        idx_top_k = _get_top_k_cm_indices(self.confusion_matrix, self.num_classes, self.top_k)

        if self.reduce == "macro":
            """ Macro-averaged f1 score is calculated by dividing the f1-score per class by the number of classes with 
            samples. Only the top_k classes regarding class frequency are considered.
            There is another definition of macro f1-score, which calculated the harmonic mean of the
            macro-weighted precision and the macro-weighted recall, which is not used here. """
            f1_score_per_class = tf.gather(f1_score_per_class, indices=idx_top_k, axis=0)
            num_classes_with_samples = _get_num_classes_with_samples(self.confusion_matrix, self.num_classes,
                                                                     self.top_k)
            self.f1score.assign(
                tf.math.divide_no_nan(tf.math.reduce_sum(f1_score_per_class),
                                      num_classes_with_samples))
        else:
            """ Always yields: micro-recall=micro-precision=micro-f1score, if top_k = num_classes """
            top_k_cm = _get_top_k_cm(self.confusion_matrix, self.num_classes, self.top_k)
            weights = tf.math.divide_no_nan(tf.reduce_sum(top_k_cm, axis=-1), tf.reduce_sum(
                top_k_cm))
            recall_per_class = tf.gather(recall_per_class, indices=idx_top_k, axis=0)
            self.f1score.assign(tf.reduce_sum(recall_per_class * weights))

        return self.f1score

    def reset_states(self):
        """
        Resets f1score. Should be called at the end of each epoch.
        @return: Nothing
        """
        self.f1score.assign(0.)
        self.confusion_matrix.assign(tf.zeros(
            shape=(self.num_classes, self.num_classes), dtype=tf.dtypes.float32, name=None
        ))


def _get_top_k_cm_indices(cm, num_classes, top_k):
    """
    @param cm: confusion matrix
    @param num_classes: number of classes in confusion matrix.
    @param top_k: top_k classes, if top_k >= 0, else bottom_k classes.
    Must be in range [-num_classes, num_classes]
    @return: indices of top_k classes with the most samples (top_k >= 0) or least samples (top_k < 0).
    """
    assert abs(top_k) <= num_classes, "Top_k must be <= abs(num_classes)"

    if top_k >= 0:
        """ top_k classes"""
        _, idx_top_k = tf.math.top_k(tf.reduce_sum(cm, axis=-1), k=top_k)
    else:
        """ bottom k_classes """
        _, idx_all = tf.math.top_k(tf.reduce_sum(cm, axis=-1), k=num_classes)
        idx_top_k = tf.slice(idx_all, [num_classes + top_k], size=[-top_k])

    return idx_top_k


def _get_num_classes_with_samples(cm, num_classes, top_k):
    """
    @param cm: confusion matrix
    @param num_classes: number of classes in confusion matrix.
    @param top_k: top_k classes, if top_k >= 0, else bottom_k classes.
    Must be in range [-num_classes, num_classes]
    @return  number of classes with actual samples within the top_k classes.
    """
    idx_top_k = _get_top_k_cm_indices(cm, num_classes, top_k)
    num_classes_with_samples = tf.math.count_nonzero(tf.math.count_nonzero(tf.gather(cm, indices=idx_top_k, axis=0),
                                                                           axis=-1),
                                                     dtype=tf.float32)
    return num_classes_with_samples


def _get_top_k_cm(cm, num_classes, top_k):
    """ Slices the confusion matrix 'cm', so only top_k classes regarding frequency are
    considered in the returned cm. If two elements are equal, the lower-index element appears first.
    @param cm: confusion matrix
    @param num_classes: number of classes in confusion matrix.
    @param top_k: top_k classes, if top_k >= 0, else bottom_k classes.
        Must be in range [-num_classes, num_classes].
    @return sliced confusion matrix.
    """

    """ Get indices of top_k classes with the most samples"""
    idx_top_k = _get_top_k_cm_indices(cm, num_classes, top_k)
    """ Slice confusion matrix, so only top_k/bottom_k classes are considered. """
    top_k_confusion_matrix = tf.gather(tf.gather(cm, indices=idx_top_k, axis=0),
                                       indices=idx_top_k, axis=-1)

    return top_k_confusion_matrix

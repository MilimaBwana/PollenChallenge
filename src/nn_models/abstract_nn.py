from abc import ABCMeta, abstractmethod
import tensorflow as tf
from nn_util import metrics, model_reset
import re
from config import config as cfg

'''
class Meta(ABCMeta):
    """ This class ensures that every subclass of AbstractNN sets the required attributes in their __init__() method.
    See https://stackoverflow.com/questions/55481355/python-abstract-class-shall-force-derived-classes-to-initialize-variable-in-in
    """
    required_attributes = []

    def __call__(self, *args, **kwargs):
        obj = super(Meta, self).__call__(*args, **kwargs)
        for attr_name in obj.required_attributes:
            if not getattr(obj, attr_name):
                raise AttributeError('required attribute (%s) not set' % attr_name)
        return obj
'''


class AbstractNN(tf.keras.Model, metaclass=ABCMeta):
    """ Abstract Class for all nn models. """

    @abstractmethod
    def __init__(self, params):
        super(AbstractNN, self).__init__()
        self.params = params

        if params['optimizer'].lower() == 'adam':
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=params['learning_rate'])
        elif params['optimizer'].lower() == 'amsgrad':
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=params['learning_rate'], amsgrad=True)
        elif params['optimizer'].lower() == 'sgd':
            self.optimizer = tf.keras.optimizers.SGD(learning_rate=params['learning_rate'], momentum=0.0)
        else:
            raise ValueError('No valid optimizer.')

        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True,
            reduction=tf.keras.losses.Reduction.NONE)

        self.params['num_classes'] = self.params['dataset'].num_classes

        """ Train metrics."""
        self.train_metrics = []
        self.train_loss = tf.keras.metrics.Mean('loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
        self.train_recall = metrics.MultiClassRecall(name='recall', num_classes=self.params['num_classes'],
                                                     reduce='macro')
        self.train_precision = metrics.MultiClassPrecision(name='precision',
                                                           num_classes=self.params['num_classes'], reduce='macro')
        self.train_f1score = metrics.MultiClassF1Score(name='f1score', num_classes=self.params['num_classes'],
                                                       reduce='macro')
        self.train_f1score_bottom11 = metrics.MultiClassF1Score(name='f1score_bottom',
                                                                num_classes=self.params['num_classes'],
                                                                reduce='macro',
                                                                top_k=-self.params['dataset'].num_bottom_classes)

        self.train_metrics.extend(
            [self.train_precision, self.train_recall, self.train_f1score, self.train_f1score_bottom11, self.train_loss,
             self.train_accuracy])

        """ Validation metrics. """
        self.val_metrics = []
        self.val_loss = tf.keras.metrics.Mean('loss')
        self.val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
        self.val_recall = metrics.MultiClassRecall(name='recall', num_classes=self.params['num_classes'],
                                                   reduce='macro')
        self.val_precision = metrics.MultiClassPrecision(name='precision', num_classes=self.params['num_classes'],
                                                         reduce='macro')
        self.val_f1score = metrics.MultiClassF1Score(name='f1score', num_classes=self.params['num_classes'],
                                                     reduce='macro')
        self.val_f1score_bottom11 = metrics.MultiClassF1Score(name='f1score_bottom',
                                                              num_classes=self.params['num_classes'],
                                                              reduce='macro',
                                                              top_k=-self.params['dataset'].num_bottom_classes)

        self.val_metrics.extend(
            [self.val_precision, self.val_recall, self.val_f1score, self.val_f1score_bottom11, self.val_loss,
             self.val_accuracy])

        """Predict metrics. """
        self.predict_metrics = []
        self.predict_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
        self.predict_recall = metrics.MultiClassRecall(name='recall', num_classes=self.params['num_classes'],
                                                       reduce='macro')
        self.predict_precision = metrics.MultiClassPrecision(name='precision',
                                                             num_classes=self.params['num_classes'], reduce='macro')
        self.predict_f1score = metrics.MultiClassF1Score(name='f1score', num_classes=self.params['num_classes'],
                                                         reduce='macro')
        self.predict_f1score_bottom11 = metrics.MultiClassF1Score(name='f1score_bottom',
                                                                  num_classes=self.params['num_classes'],
                                                                  reduce='macro',
                                                                  top_k=-self.params['dataset'].num_bottom_classes)

        self.predict_metrics.extend(
            [self.predict_precision, self.predict_recall, self.predict_f1score, self.predict_f1score_bottom11,
             self.predict_accuracy])

    @abstractmethod
    def call(self, inputs, training=None, mask=None):
        pass

    def reset(self):
        model_reset.reset_weights(self)
        model_reset.reset_optimizer(self.optimizer, self.params['learning_rate'])

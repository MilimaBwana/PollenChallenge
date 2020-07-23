import tensorflow as tf
import numpy as np


class Constant:

    def __init__(self):
        """ The learning_rate is constant."""
        pass

    def on_epoch_end(self):
        pass


class PlateauDecay:

    def __init__(self, model, monitor, min_delta=0.01, mode='max', patience=3, decay_rate=0.5):
        """ The learning_rate is decayed by the factor decay if the monitored quantity hasn't improved since 'patience'
        epochs.

        Arguments:
            model: model that is being trained
            monitor: quantity to be monitored, e.g. accuracy, precision
            min_delta: Minimum change in the monitored quantity to be qualified as
                an improvement.
            mode: One out of {"min", "max"}. Describes if the monitored quantity should be increasing ('max') or
                decreasing ('min') to be qualified as an improvement.
            patience: num of epochs with no improvement after which learning rate is decayed.
            decay_rate: factor by which the learning rate decays. Should be in the interval (0; 1).
        """
        self.model = model
        self.monitor = monitor
        self.min_delta = min_delta
        self.mode = mode
        self.patience = patience
        self.decay_rate = decay_rate
        self.wait = 0

        if mode not in ['min', 'max']:
            raise ValueError('EarlyStopping mode is not min or max.')

        if mode == 'min':
            self.monitor_op = tf.math.less
            self.min_delta *= -1
            self.best = np.Inf
        else:
            self.monitor_op = tf.math.greater
            self.best = -np.Inf

    def on_epoch_end(self):
        """Called at the end of each epoch. Checks if monitored quantity in current epoch is better than in the best
        epoch. If not, the learning_rate is decayed, if 'patience' epochs have passed since the last improvement.
        @return: Nothing."""

        current_monitor = self.monitor.result()

        if self.monitor_op(current_monitor - self.min_delta, self.best):
            """ Current epoch better """
            self.best = current_monitor
            self.wait = 0

        else:
            """ Current epoch worse """
            self.wait += 1

            if self.wait >= self.patience:
                """ Decay learning rate, if monitored quantity gets worse or is on a plateau """
                old_lr = self.model.optimizer.lr.read_value()
                self.model.optimizer.lr.assign(old_lr * self.decay_rate)
                print("Reduced learning_rate to {}".format(self.model.optimizer.lr.read_value()))
                self.wait = 0


class ExponentialDecay:

    def __init__(self, model, decay_epochs=5, decay_rate=0.5):
        """ The learning_rate is decayed by the factor 'decay_rate' every 'decay_epochs'.

        Arguments:
            model: model that is being trained
            decay_epochs: number of epochs after which the learning rate is decayed
            decay_rate: factor by which the learning rate decreases. Should be in the interval (0; 1).
        """
        self.model = model
        self.decay_epochs = decay_epochs
        self.decay_rate = decay_rate
        self.wait = 0

    def on_epoch_end(self):
        """Called at the end of each epoch. Decays the learning_rate, if decay_epochs have passed since the last decay
        or start.
        @return: Nothing."""

        self.wait += 1

        if self.wait % self.decay_epochs == 0:
            old_lr = self.model.optimizer.lr.read_value()
            self.model.optimizer.lr.assign(old_lr * self.decay_rate)
            print("Reduced learning_rate to {}".format(self.model.optimizer.lr.read_value()))


class DecayWithUnfreeze:

    def __init__(self, model, decay_epochs=5, new_lr=1e-5, unfreeze=True):
        """ The learning_rate is decayed one time after 'decay_epochs' to 'new_lr'. If 'unfreeze' is true and
        the model contains a backbone, the backbone is unfreezed after 'decay_epochs'

        Arguments:
            model: model that is being trained
            decay_epochs: number of epochs after which the learning rate is decayed
            new_lr: new learning rate after expiring 'decay_epochs'.
            unfreeze: if true, the backbone is unfreezed
        """
        self.model = model
        self.decay_epochs = decay_epochs
        self.new_lr = new_lr
        self.unfreeze = unfreeze
        self.wait = 0

    def on_epoch_end(self):
        """Called at the end of each epoch. Decays the learning_rate, if decay_epochs have passed since the last decay
        or start.
        @return: Nothing."""

        self.wait += 1

        if self.wait == self.decay_epochs:
            old_lr = self.model.optimizer.lr.read_value()
            self.model.optimizer.lr.assign(old_lr * self.new_lr)
            if hasattr(self.model, 'backbone'):
                for layer in self.backbone.layers:
                    layer.trainable = False
                print("Reduced learning_rate to {} and unfreeze backbone.".format(self.model.optimizer.lr.read_value()))
            else:
                print("Reduced learning_rate to {}. No backbone found.".format(self.model.optimizer.lr.read_value()))


import numpy as np
import tensorflow as tf


class BestEpochCheckpoint:
    """ Saves only the checkpoints for the epoch with the best monitored quantity.

    Arguments:
        model: model that is being trained
        directory: the path to a directory in which to write checkpoints
        max_to_keep: the number of checkpoints to keep.
        monitor: quantity to be monitored, e.g. accuracy, precision
        min_delta: Minimum change in the monitored quantity to be qualified as
            an improvement.
        mode: One out of {"min", "max"}. Describes if the monitored quantity should be increasing ('max') or decreasing ('min') to
          be qualified as an improvement.
    """

    def __init__(self, model, directory, max_to_keep, monitor, min_delta=0.01, mode='max'):

        self.model = model
        self.directory = directory
        self.monitor = monitor
        self.min_delta = min_delta
        self.mode = mode
        self.wait = 0

        self.ckpt = tf.train.Checkpoint(step=tf.Variable(1), net=self.model)
        self.manager = tf.train.CheckpointManager(self.ckpt, self.directory, max_to_keep=max_to_keep)

        if mode not in ['min', 'max']:
            raise ValueError('Monitoring mode must be min or max.')

        if mode == 'min':
            self.monitor_op = tf.math.less
            self.min_delta *= -1
            self.best = np.Inf
        else:
            self.monitor_op = tf.math.greater
            self.best = -np.Inf

    def on_epoch_end(self, epoch):
        """ Called at the end of each epoch. Checks if monitored quantity in current epoch is better than in the best
        epoch. If so, a new checkpoint is saved.
        @param epoch: current epoch.
        @return Nothing."""
        current_monitor = self.monitor.result()

        if self.monitor_op(current_monitor - self.min_delta, self.best):
            # Current epoch better.
            self.best = current_monitor
            self.wait = 0
            save_path = self.manager.save(checkpoint_number=epoch)
            print("Saved checkpoint for epoch {}: {}".format(epoch, save_path))

        else:
            # nCurrent epoch worse.
            self.wait += 1


class EveryEpochCheckpoint:
    """Saves a checkpoint for every epoch.

    Arguments:
        model: model that is being trained
        directory: the path to a directory in which to write checkpoints
        max_to_keep: the number of checkpoints to keep.
    """

    def __init__(self, model, directory, max_to_keep):
        self.model = model
        self.directory = directory
        self.ckpt = tf.train.Checkpoint(step=tf.Variable(1), net=self.model, optimizer=self.model.optimizer)
        self.manager = tf.train.CheckpointManager(self.ckpt, self.directory, max_to_keep=max_to_keep)

    def on_epoch_end(self, epoch):
        """ Saves checkpoint.
        @param epoch: current epoch.
        @return: Nothing."""
        save_path = self.manager.save(checkpoint_number=epoch)
        print("Saved checkpoint for epoch {}: {}".format(epoch, save_path))

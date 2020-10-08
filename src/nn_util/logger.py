import numpy as np
import tensorflow as tf
import datetime
import os
from pathlib import Path
from _collections import defaultdict
import json
from sklearn.metrics import classification_report
from nn_util import tensorboard_ops


class MetricLogger:
    """
    Class to log the metrics of the given 'model' for training, validation and prediction during a run.
    Log file is saved to 'directory'. Only the epoch with the best monitored quantity for each run is saved.
    Arguments:
        model: model that is being trained
        directory: the path to a directory in which to write the logfile.
        monitor: quantity to be monitored, e.g. accuracy, precision
        min_delta: Minimum change in the monitored quantity to be qualified as
            an improvement.
        mode: One out of {"min", "max"}. Describes if the monitored quantity should be increasing ('max') or decreasing
            ('min') to be qualified as an improvement.
    """

    def __init__(self, model, directory, monitor, min_delta=0.01, mode='max'):
        self.model = model
        file = self.model.params['name'] + '_logfile_' + datetime.datetime.now().strftime('%d.%m.%Y-%H:%M') + ".txt"
        Path(directory).mkdir(parents=True, exist_ok=True)
        self.save_path = os.path.join(directory, file)
        self.monitor = monitor
        self.min_delta = min_delta
        self.mode = mode

        if mode not in ['min', 'max']:
            raise ValueError('EarlyStopping mode is not min or max.')

        self.total_train_metrics_result = defaultdict(list)
        self.total_val_metrics_result = defaultdict(list)
        self.total_predict_metrics_result = defaultdict(list)
        self.train_metric_results = defaultdict(float)
        self.val_metric_results = defaultdict(float)
        self.best_epoch = 0
        self.total_predict_cm = tf.Variable(
            initial_value=np.zeros((self.model.params['num_classes'], self.model.params['num_classes'])))

        if self.mode == 'min':
            self.monitor_op = tf.math.less
            self.min_delta *= -1
            self.best_monitor = np.Inf

        else:
            self.monitor_op = tf.math.greater
            self.best_monitor = -np.Inf

        with open(self.save_path, "a") as writer:
            writer.write('Start: ' + datetime.datetime.now().strftime('%d.%m.%Y-%H:%M'))
            writer.write('\nParameter: \n')
            for key, value in model.params.items():
                writer.write('\t' + key + ': ' + str(value) + '\n')

    def on_epoch_end(self, epoch):
        """
        Called at the end of each epoch during training and validating.
        Checks if monitored quantity in current epoch is better than in the best
        epoch. If so, the values of every train and val metric are saved.
        @param epoch: current epoch.
        @return: Nothing.
        """

        current_monitor = self.monitor.result()

        if self.monitor_op(current_monitor - self.min_delta, self.best_monitor):
            # Current epoch better.
            self.best_epoch = epoch
            self.best_monitor = current_monitor

            for metric in self.model.train_metrics:
                self.train_metric_results[metric.name] = metric.result()

            for metric in self.model.val_metrics:
                self.val_metric_results[metric.name] = metric.result()

    def on_train_val_end(self, iteration):
        """
        Called after training and validation. Writes all metric results of the best epoch
        to the logfile and resets the metrics.
        @param iteration: current iteration.
        @return: Nothing.
        """
        with open(self.save_path, "a") as writer:
            # Write metric results to logfile.
            writer.write('\nRun: ' + str(iteration))
            writer.write('\tBest Epoch: ' + str(self.best_epoch) + '\n')
            writer.write('\tTraining: \n')
            for key, value in self.train_metric_results.items():
                self.total_train_metrics_result[key].append(value.numpy())
                writer.write('\t\t' + key + ': ' + str(value.numpy()) + '\n')

            writer.write('\tValidation: \n')
            for key, value in self.val_metric_results.items():
                self.total_val_metrics_result[key].append(value.numpy())
                writer.write('\t\t' + key + ': ' + str(value.numpy()) + '\n')

        # Reset metrics.
        self.best_epoch = 0
        self.train_metric_results = defaultdict(float)
        self.val_metric_results = defaultdict(float)

        if self.mode == 'min':
            self.best_monitor = np.Inf
        else:
            self.best_monitor = -np.Inf

    def on_predict_end(self):
        """
        Called after predicting. Writes all metric results to the logfile.
        @return: Nothing.
        """
        with open(self.save_path, "a") as writer:
            writer.write('\tPrediction: \n')
            for metric in self.model.predict_metrics:
                writer.write('\t\t' + metric.name + " :" + str(metric.result().numpy()) + '\n')
                self.total_predict_metrics_result[metric.name].append(metric.result().numpy())

    def on_run_end(self):
        """
        Called after all iterations. Writes the averaged metrics over all runs for training, validation and prediction
        to the log file.
        @return: Nothing.
        """
        with open(self.save_path, "a") as writer:
            writer.write('\nTotal result'  '\n Training and Evaluation:\n')
            writer.write('\tTraining: \n')
            for key in self.total_train_metrics_result.keys():
                total_metric_avg = np.around(sum(self.total_train_metrics_result[key]) / len(
                    self.total_train_metrics_result[key]), decimals=5)
                total_metric_stddev = np.around(np.std(self.total_train_metrics_result[key]), decimals=5)
                writer.write('\t\t' + key + ': ' + str(total_metric_avg) + ' (+- ' + str(total_metric_stddev) + ')\n')

            writer.write('\tValidation: \n')
            for key in self.total_val_metrics_result.keys():
                total_metric_avg = np.around(sum(self.total_val_metrics_result[key]) / len(
                    self.total_val_metrics_result[key]), decimals=5)
                total_metric_stddev = np.around(np.std(self.total_val_metrics_result[key]), decimals=5)
                writer.write('\t\t' + key + ': ' + str(total_metric_avg) + ' (+- ' + str(total_metric_stddev) + ')\n')

            writer.write('\tPrediction: \n')
            for key in self.total_predict_metrics_result.keys():
                total_metric_avg = np.around(sum(self.total_predict_metrics_result[key]) / len(
                    self.total_predict_metrics_result[key]), decimals=5)
                total_metric_stddev = np.around(np.std(self.total_predict_metrics_result[key]), decimals=5)
                writer.write('\t\t' + key + ': ' + str(total_metric_avg) + ' (+- ' + str(total_metric_stddev) + ')\n')
            writer.write('End: ' + datetime.datetime.now().strftime('%d.%m.%Y-%H:%M'))


class ConfusionMatrixLogger:
    """
    Class to log the averaged confusion matrix over all iterations.
    Arguments:
        model: model that is being trained
        directory: the path to a directory in which to write the logfile.
        dictionary: mapping between class names (keys) and labels(values).
    """

    def __init__(self, model, directory, dictionary):
        self.model = model
        self.total_cm = tf.Variable(tf.zeros([self.model.params['num_classes'], self.model.params['num_classes']]),
                                    dtype=tf.float32)
        self.directory = directory
        self.dictionary = dictionary
        Path(directory).mkdir(parents=True, exist_ok=True)

    def on_predict_end(self, confusion_matrix):
        """Called after predicting. Updates the confusion matrix.
        @param confusion_matrix: confusion matrix to add.
        @return: Nothing"""
        self.total_cm = tf.math.add(self.total_cm, confusion_matrix)

    def on_run_end(self):
        """
        Called after all iterations. Writes the averaged metrics over all iterations for training, validation and
        prediction.
        @return: Nothing.
        """
        predict_cm_writer = tf.summary.create_file_writer(self.directory)
        with predict_cm_writer.as_default():
            cm_image = tensorboard_ops.plot_confusion_matrix(self.total_cm,
                                                             class_names=
                                                             list(self.dictionary.keys()),
                                                             title='Confusion Matrix Predictions')
            tf.summary.image('Confusion matrix - Complete Run', cm_image,
                             max_outputs=3, step=0)


class PredictionJsonLogger:
    """
    Class to log the filenames and predicted labels in the testset. After each iteration, the filenames and predicted
    labels are saved in a json file.
    Currently not working with @tf.function due to issues with tf.TensorArray
    (https://github.com/tensorflow/tensorflow/issues/40276)
    Arguments:
        model: model that is being trained
        directory: the path to a directory in which to write the logfile.
        class_shift: used for the italian dataset. Labels given in the dataset structure there start at index 1, whereas
            model predictions start at index 0. Therefore the predictions need to be shifted up by 1.
    """

    def __init__(self, model, directory, class_shift=1):
        self.model = model
        self.directory = directory
        Path(directory).mkdir(parents=True, exist_ok=True)
        self.predictions_list = tf.TensorArray(tf.int64, size=0, dynamic_size=True, clear_after_read=True)
        self.filenames_list = tf.TensorArray(tf.string, size=0, dynamic_size=True, clear_after_read=True)
        self.index = 0
        self.class_shift = class_shift

    def on_batch_predict_end(self, filenames, predictions):
        """
        Logs the filenames and predictions after a minibatch prediction
        @param filenames: filenames to be predict onto.
        @param predictions:
        @return: Nothing
        """
        self.filenames_list = self.filenames_list.write(self.index, filenames)
        self.predictions_list = self.predictions_list.write(self.index, predictions)
        self.index += 1

    def on_predict_end(self, iteration):
        """
        Writes the (filenames, prediction)-pairs of one iteration to a json-file.
        @param iteration: current iteration
        @return: Nothing.
        """
        filenames = self.filenames_list.concat().numpy()
        predictions_list = self.predictions_list.concat().numpy()
        helper_list = []

        for filename, prediction in zip(filenames,
                                        predictions_list):
            tmp_dict = {'Filename': filename.decode("utf-8"), 'Class': str(prediction + self.class_shift)}
            helper_list.append(tmp_dict)

        with open(self.directory + '/predictions_iteration_' + str(iteration) + '.json', 'w') as file:
            json.dump(helper_list, file)

        """ Reset TensorArrays."""
        self.index = 0
        self.predictions_list.close()
        self.filenames_list.close()
        self.predictions_list = tf.TensorArray(tf.int64, size=0, dynamic_size=True, clear_after_read=True)
        self.filenames_list = tf.TensorArray(tf.string, size=0, dynamic_size=True, clear_after_read=True)


class ValidationMetricLogger:
    """
    Class to log the filenames and predicted labels in the testset. After each iteration, the filenames and predicted
    labels are saved in a json file.
    Arguments:
        model: model that is being trained
        directory: the path to a directory in which to write the logfile.
    """

    def __init__(self, model, directory, dictionary):
        self.model = model
        self.directory = directory
        self.dictionary = dictionary
        Path(directory).mkdir(parents=True, exist_ok=True)
        self.y_true_list = tf.TensorArray(tf.int64, size=0, dynamic_size=True, clear_after_read=True)
        self.y_pred_list = tf.TensorArray(tf.int64, size=0, dynamic_size=True, clear_after_read=True)
        self.index = 0

    def on_batch_val_end(self, y_pred, y_true):
        """
        Logs the labels and predictions after a minibatch val step
        @param y_pred: batch of predicted labels
        @param y_true: batch of labels
        @return: Nothing
        """
        self.y_pred_list = self.y_pred_list.write(self.index, y_pred)
        self.y_true_list = self.y_true_list.write(self.index, y_true)
        self.index += 1

    def on_epoch_end(self, epoch, iteration):
        save_path = self.directory + "/iteration_" + str(iteration) + '_val_metrics.txt'
        y_preds = self.y_pred_list.concat().numpy()
        y_true = self.y_true_list.concat().numpy()
        report = classification_report(y_true=y_true, y_pred=y_preds, target_names=list(self.dictionary.keys()),
                                       output_dict=True, zero_division=0)

        with open(save_path, 'a') as file:
            tmp_dict = {'Epoch': str(epoch), 'Metrics': report}
            file.write(json.dumps(tmp_dict) + '\n\n')

        # Reset TensorArrays.
        self.index = 0
        self.y_true_list.close()
        self.y_pred_list.close()
        self.y_true_list = tf.TensorArray(tf.int64, size=0, dynamic_size=True, clear_after_read=True)
        self.y_pred_list = tf.TensorArray(tf.int64, size=0, dynamic_size=True, clear_after_read=True)


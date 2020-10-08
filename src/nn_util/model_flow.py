import tensorflow as tf
from config import config as cfg
from config import util_ops
from preprocessing import tf_record_reader, tf_record_writer
from nn_util import metrics, weighted_gradients, tensorboard_ops, checkpoint_handler, regularization, \
    learning_rate_handler, logger
import re


@tf.function
def train_step(model, sample_batch):
    """ Executes a single train step on a batch of samples.
    @param model: model to train.
    @param sample_batch: a batch out of (image, label, filename)-tuples.
    @return Nothing."""
    images, labels, _ = sample_batch

    with tf.GradientTape() as tape:
        logits = model(images, training=True)
        per_sample_loss = model.loss_object(y_pred=logits, y_true=labels)

        if model.params['weighted_gradients'].lower() == 'inversefrequency':
            per_sample_loss = weighted_gradients.inverse_frequency_weighted_loss(per_sample_loss,
                                                                                 model.params['class_frequencies'],
                                                                                 labels)
        elif model.params['weighted_gradients'].lower() == 'betafrequency':
            per_sample_loss = weighted_gradients.beta_frequency_weighted_loss(per_sample_loss,
                                                                              model.params['class_frequencies'],
                                                                              labels,
                                                                              beta=0.9)
        elif model.params['weighted_gradients'].lower() == 'focalloss':
            per_sample_loss = weighted_gradients.focal_loss(per_sample_loss, logits=logits, labels=labels,
                                                            gamma=2)
        elif model.params['weighted_gradients'].lower() == 'weightedfocalloss':
            per_sample_loss = weighted_gradients.beta_weighted_focal_loss(per_sample_loss,
                                                                          model.params['class_frequencies'],
                                                                          logits=logits, labels=labels,
                                                                          gamma=2, beta=0.9)
        loss = tf.reduce_sum(per_sample_loss) / model.params['batch_size']

        if model.params['regularized_loss'] == 'L1':
            loss = regularization.l1_regularized_loss(loss, model, weight_decay=0.0005)
        elif model.params['regularized_loss'] == 'L2':
            loss = regularization.l2_regularized_loss(loss, model, weight_decay=0.0005)

    gradients = tape.gradient(loss, model.trainable_variables)
    model.optimizer.apply_gradients(grads_and_vars=zip(gradients, model.trainable_variables))

    if model.params['normalized_weights'].lower() == 'wvn':
        regularization.normalize_weights(model)

    model.train_loss(loss)
    model.train_accuracy.update_state(y_pred=logits, y_true=labels)
    model.train_recall.update_state(y_pred=tf.argmax(logits, axis=-1), y_true=labels)
    model.train_precision.update_state(y_pred=tf.argmax(logits, axis=-1), y_true=labels)
    model.train_f1score.update_state(y_pred=tf.argmax(logits, axis=-1), y_true=labels)
    model.train_f1score_bottom11.update_state(y_pred=tf.argmax(logits, axis=-1), y_true=labels)


@tf.function
def val_step(model, sample_batch):
    """ Executes a single validation step on a batch of samples.
    @param model: model to valid.
    @param sample_batch: a batch out of (image, label, filename)-tuples.
    @return Nothing.
    """
    images, labels, _ = sample_batch
    logits = model(images, training=False)  # Call model.call function

    per_sample_loss = model.loss_object(y_pred=logits, y_true=labels)

    if model.params['weighted_gradients'].lower() == 'inversefrequency':
        per_sample_loss = weighted_gradients.inverse_frequency_weighted_loss(per_sample_loss,
                                                                             model.params['class_frequencies'], labels)
    elif model.params['weighted_gradients'].lower() == 'betafrequency':
        per_sample_loss = weighted_gradients.beta_frequency_weighted_loss(per_sample_loss,
                                                                          model.params['class_frequencies'],
                                                                          labels,
                                                                          beta=0.9)
    elif model.params['weighted_gradients'].lower() == 'focalloss':
        per_sample_loss = weighted_gradients.focal_loss(per_sample_loss, logits=logits, labels=labels,
                                                        gamma=2)
    elif model.params['weighted_gradients'].lower() == 'weightedfocalloss':
        per_sample_loss = weighted_gradients.beta_weighted_focal_loss(per_sample_loss,
                                                                      model.params['class_frequencies'],
                                                                      logits=logits, labels=labels,
                                                                      gamma=2, beta=0.9)

    loss = tf.reduce_sum(per_sample_loss) / model.params['batch_size']

    if model.params['regularized_loss'] == 'L1':
        loss = regularization.l1_regularized_loss(loss, model, weight_decay=0.0005)
    elif model.params['regularized_loss'] == 'L2':
        loss = regularization.l2_regularized_loss(loss, model, weight_decay=0.0005)

    model.val_loss(loss)
    model.val_accuracy.update_state(y_pred=logits, y_true=labels)
    model.val_recall.update_state(y_pred=tf.argmax(logits, axis=-1), y_true=labels)
    model.val_precision.update_state(y_pred=tf.argmax(logits, axis=-1), y_true=labels)
    model.val_f1score.update_state(y_pred=tf.argmax(logits, axis=-1), y_true=labels)
    model.val_f1score_bottom11.update_state(y_pred=tf.argmax(logits, axis=-1), y_true=labels)

    #validation_logger.on_batch_val_end(y_pred=tf.argmax(logits, axis=-1), y_true=labels)


# @tf.function
def predict_step(model, sample_batch, prediction_logger):
    """ Executes a single predicts step on a batch of samples.
    @param model: model to predict onto.
    @param sample_batch: a batch out of (image, label, filename)-tuples.
    @param prediction_logger: logger to write filenames with predicted labels to a json file.
    @return Nothing."""
    images, labels, filenames = sample_batch
    logits = model(images, training=False)  # Call model.call function

    model.predict_accuracy.update_state(y_pred=logits, y_true=labels)
    model.predict_recall.update_state(y_pred=tf.argmax(logits, axis=-1), y_true=labels)
    model.predict_precision.update_state(y_pred=tf.argmax(logits, axis=-1), y_true=labels)
    model.predict_f1score.update_state(y_pred=tf.argmax(logits, axis=-1), y_true=labels)
    model.predict_f1score_bottom11.update_state(y_pred=tf.argmax(logits, axis=-1), y_true=labels)

    prediction_logger.on_batch_predict_end(filenames, tf.argmax(logits, axis=-1))


def train_val_predict(model, debugging=False):
    """ Trains and evaluates the given model on ds_train/ ds_val and evaluates it on ds_test
    for model.params['iterations'] times.
    @param model: model to train, evaluate and test.
    @param debugging: if true, script is running in debug mode.
    @return: Nothing
    """

    # Load class frequencies for weighted gradients.
    model.params['class_frequencies'] = weighted_gradients.create_frequency_dict(model.params['dataset'].count_json)

    ds_train = tf_record_reader.read(tf.estimator.ModeKeys.TRAIN, model.params)
    ds_val = tf_record_reader.read(tf.estimator.ModeKeys.EVAL, model.params)
    ds_test = tf_record_reader.read(tf.estimator.ModeKeys.PREDICT, model.params)

    # Define log directories for tensorboard and checkpoints
    log_dir = cfg.LOG_DIR + '/' + model.params['name']
    train_log_dir = log_dir + '/train/'
    val_log_dir = log_dir + '/val/'
    predict_log_dir = log_dir + '/predict/'
    checkpoints_dir = cfg.CHKPT_DIR + '/' + model.params['name']

    # Clear log folders
    util_ops.clear_directory(train_log_dir, clear_subdirectories=True)
    util_ops.clear_directory(val_log_dir, clear_subdirectories=True)
    util_ops.clear_directory(checkpoints_dir)
    util_ops.clear_directory(predict_log_dir, clear_subdirectories=True)

    # Define logger across all iterations.
    metric_logger = logger.MetricLogger(model, log_dir, model.val_f1score, min_delta=0.005, mode='max')
    cm_logger = logger.ConfusionMatrixLogger(model, predict_log_dir + 'images', model.params['dataset'].reverse_dictionary)

    #if re.match('[a-zA-Z]+_15$', model.params['dataset'].name) or re.match('[a-zA-Z]+_31$', model.params['dataset'].name):
    prediction_logger = logger.PredictionJsonLogger(model, predict_log_dir + 'json', class_shift=0)
    #validation_logger = logger.ValidationMetricLogger(model, val_log_dir + 'metrics', model.params['dataset'].dictionary)
    #elif re.match('[a-zA-Z]+_4$', model.params['dataset'].name):
    #    prediction_logger = logger.PredictionJsonLogger(model, predict_log_dir + 'json', class_shift=1)

    for iteration in range(1, model.params['iterations'] + 1):
        print('--------------------------------\nIteration ' + str(iteration) + '\n--------------------------------')

        train_val(model=model, debugging=debugging, ds_train=ds_train, ds_val=ds_val, train_log_dir=train_log_dir,
                  val_log_dir=val_log_dir, checkpoints_dir=checkpoints_dir, metric_logger=metric_logger,
                  iteration=iteration)

        predict(model=model, debugging=debugging, ds_test=ds_test, predict_log_dir=predict_log_dir,
                checkpoints_dir=checkpoints_dir,
                metric_logger=metric_logger, cm_logger=cm_logger, prediction_logger=prediction_logger,
                iteration=iteration)

        util_ops.clear_directory(checkpoints_dir)
        model.reset()

    cm_logger.on_run_end()
    metric_logger.on_run_end()


def train_val(model, debugging, ds_train, ds_val, train_log_dir, val_log_dir, checkpoints_dir, metric_logger,
              iteration=None):
    """
    Trains and validates the model for single iteration.
    @param model: model to train and validate
    @param debugging: if true, script is running in debug mode.
    @param ds_train: train TFRecordDataset.
    @param ds_val: validation TFRecordDataset.
    @param train_log_dir: directory to write log files for training.
    @param val_log_dir: directory to write log files for validation.
    @param checkpoints_dir: directory save model checkpoints.
    @param metric_logger: logger to keep track of model performance.
    @param iteration: current iteration number.
    @return: Nothing
    """
    if iteration:
        train_log_dir = train_log_dir + "iteration_" + str(iteration) + '/'
        val_log_dir = val_log_dir + "iteration_" + str(iteration) + '/'

    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    val_summary_writer = tf.summary.create_file_writer(val_log_dir)

    # Define checkpoint handler: best_checkpoint.
    ckpt_handler = checkpoint_handler.BestEpochCheckpoint(model, checkpoints_dir, 10, model.val_f1score,
                                                          min_delta=0.005, mode='max')

    # Define learning rate strategy.
    if model.params['lr_strategy'].lower() == 'plateaudecay':
        lr_handler = learning_rate_handler.PlateauDecay(model, model.val_f1score, min_delta=0.005, patience=4,
                                                        mode='max',
                                                        decay_rate=0.5)
    elif model.params['lr_strategy'].lower() == 'exponentialdecay':
        lr_handler = learning_rate_handler.ExponentialDecay(model, decay_epochs=5,
                                                            decay_rate=0.5)
    elif model.params['lr_strategy'].lower() == 'decaywithunfreeze':
        lr_handler = learning_rate_handler.DecayWithUnfreeze(model, decay_epochs=5,
                                                             new_lr=model.params['learning_rate'] / 2, unfreeze=True)
    else:
        lr_handler = learning_rate_handler.Constant()

    for epoch in range(1, model.params['epochs'] + 1):
        print('Epoch:' + str(epoch))

        # Training
        sample_nr = 0
        for sample in ds_train:
            train_step(model, sample)

            if debugging and sample_nr == 1:  # Testing
                break

            sample_nr += 1

        if not debugging:
            with train_summary_writer.as_default():
                for metric in model.train_metrics:
                    tf.summary.scalar(metric.name, metric.result(), step=epoch)

        # Validation
        sample_nr = 0
        for sample in ds_val:

            val_step(model, sample)
            if debugging and sample_nr == 1:  # Testing
                break
            sample_nr += 1

        if not debugging:
            with val_summary_writer.as_default():
                for metric in model.val_metrics:
                    tf.summary.scalar(metric.name, metric.result(), step=epoch)

        template = 'Epoch {}, Loss: {} , Acc: {}, Precision: {}, Recall: {}, F1-score: {}, F1(bottomClasses): {} \n ' \
                   'ValLoss: {}, ValAcc: {}, ValPrecision:{}, ValRecall:{}, ValF1-score: {}, ValF1(bottomClasses): {}'
        print(template.format(epoch,
                              model.train_loss.result(),
                              model.train_accuracy.result(),
                              model.train_precision.result(),
                              model.train_recall.result(),
                              model.train_f1score.result(),
                              model.train_f1score_bottom11.result(),
                              model.val_loss.result(),
                              model.val_accuracy.result(),
                              model.val_precision.result(),
                              model.val_recall.result(),
                              model.val_f1score.result(),
                              model.val_f1score_bottom11.result()))

        ckpt_handler.on_epoch_end(epoch)
        lr_handler.on_epoch_end()
        metric_logger.on_epoch_end(epoch)

        for metric in model.train_metrics:
            metric.reset_states()

        for metric in model.val_metrics:
            metric.reset_states()

    if model.params['normalized_weights'].lower() == 'wvn':
        regularization.rescale_weights_classification_layer(model, model.params['class_frequencies'], gamma=0.3)
    metric_logger.on_train_val_end(iteration)


#@tf.function
def predict(model, debugging, ds_test, predict_log_dir, checkpoints_dir, metric_logger, cm_logger, prediction_logger,
            iteration=None):
    """
    Uses a pretrained model to predict classes on the 'ds_test''. Model is restored by loading latest checkpoint.
    @param model: model to test.
    @param debugging: if true, script is running in debug mode.
    @param ds_test: test TFRecordDataset.
    @param predict_log_dir: directory to write log files for testing.
    @param checkpoints_dir: directory to load trained model checkpoints.
    @param metric_logger: logger to keep track of model performance.
    @param cm_logger: logger to keep track of the confusion matrix for all predictions.
    @param prediction_logger: logger to write filenames with predicted labels to a json file.
    @param iteration: current iteration number.
    @return:
    """
    if iteration:
        predict_log_dir = predict_log_dir + "iteration_" + str(iteration) + '/'

    predict_log_dir_metrics = predict_log_dir + 'metrics'

    # Load pretrained model.
    checkpoint = tf.train.Checkpoint(net=model)
    status = checkpoint.restore(tf.train.latest_checkpoint(checkpoints_dir))
    status.assert_existing_objects_matched()
    status.expect_partial()

    predict_summary_writer = tf.summary.create_file_writer(predict_log_dir_metrics)
    print('Prediction')
    sample_nr = 0

    for sample in ds_test:
        predict_step(model, sample, prediction_logger)
        if debugging and sample_nr == 1:  # Testing
            break
        sample_nr += 1

    with predict_summary_writer.as_default():
        for metric in model.predict_metrics:
            tf.summary.scalar(metric.name, metric.result(), step=1)

    metric_logger.on_predict_end()
    cm_logger.on_predict_end(model.predict_recall.get_confusion_matrix())
    # TODO: @tf.function compatible, currently not possible: https://github.com/tensorflow/tensorflow/issues/40276
    prediction_logger.on_predict_end(iteration)

    template = 'Precision: {}, Recall: {}, F1-score: {}, F1-Score(bottomClasses): {}'
    print(template.format(
        model.predict_precision.result(),
        model.predict_recall.result(),
        model.predict_f1score.result(),
        model.predict_f1score_bottom11.result()))

    for metric in model.predict_metrics:
        metric.reset_states()

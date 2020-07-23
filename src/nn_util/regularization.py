import tensorflow as tf


def l2_regularized_loss(loss, model, weight_decay=0.0005):
    """
    The l2 norm of the weights (not including biases) of 'model' (multiplied by coefficient 'weight_decay') is added
    into 'loss' as a penalty term to be minimized. Function is recursive.
    @param loss: reduced loss for a mini-batch.
    @param model:
    @param weight_decay: weight decay coefficient
    @return: loss with weight penalties included.
    """
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            loss = l2_regularized_loss(loss, layer, weight_decay)
            continue
        if hasattr(layer, 'kernel'):
            loss += tf.nn.l2_loss(layer.kernel) * weight_decay

    return loss


def l1_regularized_loss(loss, model, weight_decay=0.0005):
    """
    The l1 norm of the weights (not including biases) of 'model' (multiplied by coefficient 'weight_decay') is added
    into 'loss' as a penalty term to be minimized.
    #TODO: seems to create huge regularization terms
    @param loss: reduced loss for a mini-batch.
    @param model:
    @param weight_decay: weight decay coefficient
    @return: loss with weight penalties included.
    """
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            loss = l1_regularized_loss(loss, layer, weight_decay)
            continue
        if hasattr(layer, 'kernel'):
            loss += tf.reduce_sum(tf.math.abs(layer.kernel)/2) * weight_decay

    return loss


def rescale_weights_classification_layer(model, frequency_dict, gamma=0.3):
    """ Rescales weight vectors of last layer after complete training by a factor (n_max / n_i)^gamma for a class i,
    where n is the number of samples in that class.
    @param model
    @param frequency_dict: tf.StaticHashTable with class label(keys) and class frequency (label)
    @param gamma: hyperparameter to tune effect of rescaling. A larger value of gamma leads to infrequent classes
        covering more feature space. If gamma = 0, no rescaling is applied.
    @return Nothing."""

    max_frequency = tf.math.reduce_max(frequency_dict.export()[1])
    """ Possible not ordered keys in frequency_dict should be ordered.. """
    ordered_frequencies = tf.gather(frequency_dict.export()[1], indices=tf.argsort(frequency_dict.export()[0]))
    rescale_factor = tf.math.pow(tf.math.divide_no_nan(max_frequency, ordered_frequencies),
                                 gamma)  # Shape (num_classes)
    rescale_factor = tf.broadcast_to(rescale_factor,
                                     tf.shape(model.layers[-1:][0].kernel))  # Shape (num_features, num_classes)
    model.layers[-1:][0].kernel.assign(tf.multiply(model.layers[-1:][0].kernel, rescale_factor))


def normalize_weights(model):
    """
    Normalizes weight vectors of last layer after each batch.
    @param model
    @return Nothing.
    """

    kernel_last_layer = model.layers[-1:][0].kernel
    norm = tf.broadcast_to(input=tf.norm(kernel_last_layer, ord=2, axis=0), shape=tf.shape(kernel_last_layer))
    model.layers[-1:][0].kernel.assign(tf.math.divide(model.layers[-1:][0].kernel, norm))

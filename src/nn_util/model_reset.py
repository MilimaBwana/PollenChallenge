import tensorflow as tf


def reset_weights(model):
    """
    Re-initializes the weights of each layer in the model.
    See https://github.com/keras-team/keras/issues/341
    @param model:
    @return Nothing
    """
    print('Reset weights...')

    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            reset_weights(layer)
            continue
        for k, initializer in layer.__dict__.items():
            if "initializer" not in k:
                continue
            # find the corresponding variable
            var = getattr(layer, k.replace("_initializer", ""))
            if var is not None:
                var.assign(initializer(var.shape, var.dtype))


def reset_optimizer(optimizer, learning_rate):
    """
    Re-initializes the given optimizer and the learning_rate to the initial learning rate.
    @param optimizer: used optimizer
    @param learning_rate: initial learning_rate
    """
    print('Reset optimizer...')
    for var in optimizer.variables():
        var.assign(tf.zeros_like(var))
    optimizer.lr.assign(learning_rate)



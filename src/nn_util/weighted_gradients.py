from collections import defaultdict
import json
import tensorflow as tf


def create_frequency_dict(count_classes_json):
    """
    Calculates class specific weights out of the given json-file 'count_classes_json'.
    @param count_classes_json: json file containing dictionary with keys as class labels and class frequencies as keys.
    @return: a tf.StaticHashTable with the classes as the keys and the frequencies as values.
    """
    with open(count_classes_json, 'r') as file:
        count_classes_dict = json.load(file)

    freq = defaultdict()
    for key, value in count_classes_dict.items():
        """ Cast key to int. """
        freq[int(key)] = value

    frequency_dict = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(keys=list(freq.keys()), values=list(freq.values()),
                                            key_dtype=tf.int32,
                                            value_dtype=tf.float32), -1)

    return frequency_dict


def inverse_frequency_weighted_loss(per_sample_loss, frequency_dict, labels):
    """
    Weights the given 'per_sample_loss' by a class-specific weight.
    A class-specific weight (alpha) for a class c is the inverse of the ratio of the samples of the class c to the
    majority class samples.
    @param per_sample_loss: non-reduced loss for a mini-batch.
    @param frequency_dict: a tf.StaticHashTable with the classes as the keys and the frequencies as values.
    @param labels: labels of the minibatch.
    @return non-reduced loss and weighted loss for a minibatch.
    """

    max_frequency = tf.math.reduce_max(frequency_dict.export()[1])
    weights = tf.map_fn(
        lambda x: tf.math.divide_no_nan(max_frequency, frequency_dict.lookup(tf.cast(x, dtype=tf.int32))),
        elems=labels,
        dtype=tf.float32)
    per_sample_loss = tf.compat.v1.losses.compute_weighted_loss(
        per_sample_loss, weights=weights, reduction=tf.keras.losses.Reduction.NONE
    )

    return per_sample_loss


def beta_frequency_weighted_loss(per_sample_loss, frequency_dict, labels, beta=0.9):
    """
    Weights the given 'per_sample_loss' by a class-specific weight.
    The class specific weight of a class c can be written by (1- beta) / (1- beta^(f_c)), where f_c is the ratio of
    samples in the class c divided by the samples in the majority class.

    @param per_sample_loss: non-reduced loss for a mini-batch.
    @param frequency_dict: a tf.StaticHashTable with the classes as the keys and the frequencies as values.
    @param labels: labels of the minibatch.
    @param beta: hyperparameter, is in range [0,1). The closer beta is to 1, the closer the weights approach the inverse
        frequency of the class c.
    @return non-reduced loss and weighted loss for a minibatch.
    """
    max_frequency = tf.math.reduce_max(frequency_dict.export()[1])
    weights = tf.map_fn(
        lambda x: tf.math.divide_no_nan(1 - beta, 1 - tf.math.pow(beta, tf.math.divide_no_nan(frequency_dict.lookup(
            tf.cast(x, dtype=tf.int32)), max_frequency))),
        elems=labels,
        dtype=tf.float32)
    per_sample_loss = tf.compat.v1.losses.compute_weighted_loss(
        per_sample_loss, weights=weights, reduction=tf.keras.losses.Reduction.NONE
    )

    return per_sample_loss


def focal_loss(per_sample_loss, logits, labels, gamma=2):
    """
    Focal loss is a derivate of crossentropy. False classified samples are weighted more than correct classified
    samples. The loss for a sample is FL(p_t)=-(1-p_t)^{gamma}ln(p_t), where -ln(p_t) is the normal
    crossentropy and (1-p_t)^{gamma} is the focal_weight.
    @param per_sample_loss: non-reduced loss for a mini-batch
    @param logits: prediction probabilities for the mini-batch
    @param labels: labels of the minibatch.
    @param gamma: hyperparameter. A higher value of gamma reduces the loss contribution from easy examples and
        extends the range in which an example receives low loss. If gamma=0, focal loss = cross entropy loss.
    @return non-reduced focal loss of the mini-batch.
    """

    y_pred = tf.argmax(logits, axis=-1)
    # alpha_factor = tf.multiply(tf.ones_like(per_sample_loss, dtype=None, name=None), alpha)
    # alpha_factor = tf.where(tf.math.equal(labels, y_pred), alpha_factor, 1 - alpha_factor)

    focal_weight = tf.where(tf.math.equal(y_pred, labels), 1 - tf.reduce_max(logits, axis=-1),
                            tf.reduce_max(logits, axis=-1))
    focal_weight = tf.math.pow(focal_weight, gamma)

    return tf.multiply(per_sample_loss, focal_weight)


def beta_weighted_focal_loss(per_sample_loss, frequency_dict, logits, labels, gamma=2, beta=0.9):
    """
    Focal loss is a derivate of crossentropy. False classified samples are weighted more than correct classified
    samples. The loss for a sample is FL(p_t)=-alpha(1-p_t)^{gamma}ln(p_t), where -ln(p_t) is the normal
    crossentropy and (1-p_t)^{gamma} is the focal_weight. The class specific weight alpha of a class c can be
    written by (1- beta) / (1- beta^(f_c)), where f_c is the ratio of
    samples in the class c divided by the samples in the majority class.
    @param per_sample_loss: non-reduced loss for a mini-batch.
    @param frequency_dict: a tf.StaticHashTable with the classes as the keys and the frequencies as values.
    @param logits: prediction probabilities for the mini-batch.
    @param labels: labels of the minibatch.
    @param gamma: hyperparameter. A higher value of gamma reduces the loss contribution from easy examples and
        extends the range in which an example receives low loss. If gamma=0, focal loss = cross entropy loss.
    @param beta: hyperparameter, is in range [0,1). The closer beta is to 1, the closer the weights approach the inverse
        frequency of the class c.
    @return non reduced beta weighted focal loss of the mini-batch.
    """

    y_pred = tf.argmax(logits, axis=-1)
    max_frequency = tf.math.reduce_max(frequency_dict.export()[1])
    alpha_factor = tf.map_fn(
        lambda x: tf.math.divide_no_nan(1 - beta, 1 - tf.math.pow(beta, tf.math.divide_no_nan(frequency_dict.lookup(
            tf.cast(x, dtype=tf.int32)), max_frequency))),
        elems=labels,
        dtype=tf.float32)

    focal_weight = tf.where(tf.math.equal(y_pred, labels), 1 - tf.reduce_max(logits, axis=-1),
                            tf.reduce_max(logits, axis=-1))
    focal_weight = tf.multiply(alpha_factor, tf.math.pow(focal_weight, gamma))

    return tf.multiply(per_sample_loss, focal_weight)

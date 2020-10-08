import itertools
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import io


def plot_confusion_matrix(cm, class_names, title='Confusion matrix'):

    plot = _matplotlib_confusion_matrix(cm, class_names, title)
    plot = _plot_to_image(plot)
    return plot


def _matplotlib_confusion_matrix(cm, class_names, title='Confusion matrix'):
    """ Creates a matplotlib figure out of the given tf.math.confusion_matrix 'cm'.
    @param cm: confusion matrix tensor
    @param class_names: name of the classes.
    @return matplotlib figure of confusion matrix. """
    cm = cm.numpy() # Cast Tensor to numpy array
    with np.errstate(divide='ignore', invalid='ignore'):
        cm = np.around(np.nan_to_num(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]), decimals=2)
    cm = np.around(cm * 100).astype(np.int32)
    threshold = cm.max() * 3. / 4
    size = np.round(np.shape(cm)[0] * 0.5,0)
    figure = plt.figure(figsize=(size, size))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Use white text if squares are dark; otherwise black text.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = 'white' if cm[i, j] > threshold else 'black'
        plt.text(j, i, cm[i, j], horizontalalignment='center', color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    return figure


def _plot_to_image(figure):
    """ Converts the  figure to a PNG image and returns it. The supplied figure is closes
    and inacessible after this call.
    @param figure: matplotlib plot.
    @return PNG image of matplotlib figure."""

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly.
    # Note: Debugging delivers empty buffer.
    plt.close(figure)
    # Set buffer pointer to start.
    buf.seek(0)
    # Convert the PNG Buffer to a TF Image.
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    buf.close()
    # Add the batch dimension.
    image = tf.expand_dims(image, 0)
    return image



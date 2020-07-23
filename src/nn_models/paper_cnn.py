import tensorflow as tf
import sys
import os

""" Syspath needs to include parent directory "pollen classification" and "Code" to find sibling 
modules and database."""
file_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + "/"
sys.path.append(file_path)

from config import util_ops, dataset_config
from nn_models import abstract_nn, argparser
from nn_util import model_flow, model_reset


def main():
    """
    Parameter:
        name: Name of the model. Used for tensorboard and logfile directories.
        dataset_name: 'original_[4|15|31]', 'upsample_[4|15|31]' or 'downsample_[15|31]'. The number determines the
            number of classes. Currently 15, 31 (Augsburg dataset) or 4 (italian dataset) classes are allowed.
        input_shape: shape of the inputted image, does not include batch dimension. If the channel dimension is 3,
            RGB images are used, if last dimension is 1, grayscale images are used.
        learning_rate: initial learning rate.
        lr_strategy: adjust the learning rate during training. Can be 'PlateauDecay', 'ExponentialDecay' or 'Constant'.
        batch_size: size of combined consecutive elements.
        dropout: Float between 0 and 1. Fraction of the input units to drop.
        colormap: 'None' or 'viridis'. Only valid if 3 channels in input_shape.
        augmentation: if True, the inputted images are augmented with the given augmentation_techniques.
            TODO: currently only True, if set as command line argument.
        augmentation_techniques: flip_left_right, flip_up_down, crop, rotate, distort_color, noise. Needs to be a list.
        weighted_gradients: Recommended if dataset.py is unbalanced. Can be 'BetaFrequency', 'InverseFrequency', 'FocalLoss'
            'WeightedFocalLoss' or 'None'.
        normalized_weights: Strategy from Kim2020. Can be 'WVN' (Weight Vector Normalization) or 'None'.
        regularized_loss:  The weights of the model (multiplied by coefficient 'weight_decay') is added into the loss
            as a penalty term to be minimized. Regularization term is calculated with 'L1' or 'L2'-norm.
        epochs: number of epochs.
        iterations: number of iterations.
        num_classes: Number of classes. Is derived from dataset.py.
    """
    params = {'name': 'PaperCNN',
              'dataset_name': 'original_15',
              'input_shape': (256, 256, 1),
              'learning_rate': 5e-4,
              'lr_strategy': 'Constant',
              'optimizer': 'Adam',
              'batch_size': 64,
              'dropout': 0.25,
              'colormap': 'None',
              'augmentation': False,
              'augmentation_techniques': ['flip_left_right', 'flip_up_down', 'noise', 'crop', ],
              'weighted_gradients': 'InverseFrequency',
              'normalized_weights': 'None',
              'regularized_loss': 'L2',
              'epochs': 5,
              'iterations': 2
              }

    params = argparser.argparse_to_params(params)
    params['dataset'] = dataset_config.Dataset(params['dataset_name'])
    debugging = util_ops.is_debugging()
    model = PaperCNN(params)
    model_flow.train_val_predict(model, debugging)


class PaperCNN(abstract_nn.AbstractNN):

    def __init__(self, params):
        super(PaperCNN, self).__init__(params)
        self.params = params
        self.conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(4, 4), padding='same', activation='relu',
                                            input_shape=self.params['input_shape'], data_format='channels_last')
        self.pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same')
        self.conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=(4, 4), padding='same', activation='relu')
        self.pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same')
        self.conv3 = tf.keras.layers.Conv2D(filters=64, kernel_size=(4, 4), padding='same', activation='relu')
        self.pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same')
        # self.conv4 = tf.keras.layers.Conv2D(filters=64, kernel_size=(4, 4), padding='same', activation='relu')
        # self.pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same')

        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(units=1024, activation='relu')
        self.dropout = tf.keras.layers.Dropout(rate=self.params['dropout'])
        self.out_layer = tf.keras.layers.Dense(units=self.params['num_classes'], activation='softmax')

    def call(self, inputs, training=None, mask=None):
        """ Feeds-forward the given inputs through the model.
        Inputs is a batch of images. """
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        # x = self.conv4(x)
        # x = self.pool4(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout(x, training)
        x = self.out_layer(x)
        return x


if __name__ == "__main__":
    main()

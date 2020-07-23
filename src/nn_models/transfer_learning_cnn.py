import tensorflow as tf
from tensorflow import keras
import sys
import os

""" Syspath needs to include parent directory "pollen classification" and "Code" to find sibling 
modules and database."""
file_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + "/"
sys.path.append(file_path)

from config import config as cfg
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
        backbone: used pretrained model. Can be 'vgg16', 'densenet', 'resnet' or 'inception'.
        freeze: Percentage of layers which weights will be frozen during training. 0 means no freezing, 1 is complete
            unfreezing. The layers are unfrozen from the back to the front.
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
    params = {'name': 'TransferLearningCNN_4',
              'dataset_name': 'original_4',
              'input_shape': (256, 256, 3),
              'learning_rate': 1e-4,
              'optimizer': 'Adam',
              'lr_strategy': 'PlateauDecay',
              'batch_size': 32,
              'backbone': 'resnet',
              'freeze': 0.3,
              'dropout': 0.5,
              'colormap': 'None',
              'augmentation': False,
              'augmentation_techniques': ['crop', 'flip_left_right', 'flip_up_down', 'rotate', 'noise'],
              'weighted_gradients': 'focalloss',
              'normalized_weights': 'WVN',
              'regularized_loss': 'L2',
              'epochs': 5,
              'iterations': 10
              }

    params['dataset'] = dataset_config.Dataset(params['dataset_name'])
    params = argparser.argparse_to_params(params)
    debugging = util_ops.is_debugging()
    model = TransferLearningCNN(params)
    model_flow.train_val_predict(model, debugging)


class TransferLearningCNN(abstract_nn.AbstractNN):

    def __init__(self, params):
        super(TransferLearningCNN, self).__init__(params)
        self.params = params

        if self.params['backbone'].lower() == 'vgg16':
            self.backbone = keras.applications.vgg16.VGG16(
                input_shape=self.params['input_shape'],
                include_top=False,
                weights='imagenet',
                pooling='avg')
        elif self.params['backbone'].lower() == 'inception':
            self.backbone = keras.applications.InceptionV3(
                input_shape=self.params['input_shape'],
                include_top=False,
                weights='imagenet',
                pooling='avg')
        elif self.params['backbone'].lower() == 'resnet':
            self.backbone = keras.applications.ResNet50(
                input_shape=self.params['input_shape'],
                include_top=False,
                weights='imagenet',
                pooling='avg')
        elif self.params['backbone'].lower() == 'resnetv2':
            self.backbone = tf.keras.applications.ResNet101V2(
                input_shape=self.params['input_shape'],
                include_top=False,
                weights='imagenet',
                pooling='avg')
        elif self.params['backbone'].lower() == 'densenet':
            self.backbone = keras.applications.DenseNet121(
                input_shape=self.params['input_shape'],
                include_top=False,
                weights='imagenet',
                pooling='avg')
        else:
            raise ValueError('No valid pretrained model.')

        """Save backbone weights for reloading after an iteration. """
        self.backbone_weight_file = cfg.CHKPT_DIR + '/backbone/' + self.params['backbone'].lower() + '/backbone_weights'
        self.backbone.save_weights(self.backbone_weight_file)

        """ Backbone freezing. """
        for layer in self.backbone.layers[:int(self.params['freeze'] * len(self.backbone.layers))]:
            layer.trainable = False

        #self.dense1 = tf.keras.layers.Dense(units=1024, activation='relu')
        self.dropout1 = tf.keras.layers.Dropout(rate=self.params['dropout'])
        #self.dense2 = tf.keras.layers.Dense(units=1024, activation='relu')
        #self.dropout2 = tf.keras.layers.Dropout(rate=self.params['dropout'])
        self.out_layer = tf.keras.layers.Dense(units=self.params['num_classes'], activation='softmax')

    def call(self, inputs, training=None, mask=None):
        """ Feeds-forward the given inputs through the model.explainer = lime_image.LimeImageExplainer()
        Inputs is a batch of images. """
        x = self.backbone(inputs)
        #x = self.dense1(x)
        x = self.dropout1(x, training=training)
        #x = self.dense2(x)
        #x = self.dropout2(x, training=training)
        x = self.out_layer(x)
        return x

    def reset(self):
        """
        Resets the weights and optimizer. Original backbone weights are loaded afterwards.
        """
        super().reset()
        self.backbone.load_weights(self.backbone_weight_file)


if __name__ == "__main__":
    main()

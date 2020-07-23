import argparse


def restricted_float(x):
    """ Type for argparse argument, which only allows float values in the range [0,1]. """
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError("%r not a floating-point literal" % (x,))

    if x < 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError("%r not in range [0.0, 1.0]" % (x,))
    return x


def argparse_to_params(params):
    """
    Overwrites 'params' dictionary with set command line arguments.
    """

    parser = argparse.ArgumentParser(description='Model parameter.')
    parser.add_argument('--name', type=str, help='Name of the model.')
    parser.add_argument('--dataset_name', type=str,
                        help='Used Dataset. original_[4|15|31], downsample_[15|31] or upsample_[4|15|31], '
                             'e.g. original_15')
    parser.add_argument('--input_shape', nargs='+', type=int, help='Input shape of images.')
    parser.add_argument('--learning_rate', type=float, help='Initial learning rate.')
    parser.add_argument('--optimizer', type=str, help='Used Optimizer. Can be Adam, Amsgrad or SGD.')
    parser.add_argument('--lr_strategy', type=str, help='Constant, PlateauDecay or ExponentialDecay.')
    parser.add_argument('--batch_size', type=int, help='Batch size.')
    parser.add_argument('--dropout', type=restricted_float, help='Fraction of units to be dropped out.')
    parser.add_argument('--backbone', type=str, help='resnet, resnetv2, inception, densenet or vgg16.')
    parser.add_argument('--freeze', type=restricted_float, help='Fraction of layers in backbone model to be frozen.')
    parser.add_argument('--colormap', type=str, help='viridis or None')
    parser.add_argument('-augmentation', action='store_true', help='if set, images are augmented.')
    parser.add_argument('--augmentation_techniques', nargs='*', type=str, help='List of augmentation techniques')
    parser.add_argument('--weighted_gradients', type=str,
                        help='BetaFrequency, InverseFrequency, FocalLoss, WeightedFocalLoss or None.')
    parser.add_argument('--normalized_weights', type=str, help='WVN or None.')
    parser.add_argument('--regularized_loss', type=str, help='L1, L2 or None')
    parser.add_argument('--epochs', type=int, help='Number of epochs.')
    parser.add_argument('--iterations', type=int, help='Number of iterations.')

    args = parser.parse_args()

    """ --arguments: Read argument value"""
    if args.name:
        params['name'] = args.name
    if args.dataset_name:
        params['dataset_name'] = args.dataset_name.lower()
    if args.input_shape:
        params['input_shape'] = tuple(args.input_shape)
    if args.learning_rate:
        params['learning_rate'] = args.learning_rate
    if args.optimizer:
        params['optimizer'] = args.optimizer
    if args.lr_strategy:
        params['lr_strategy'] = args.lr_strategy
    if args.batch_size:
        params['batch_size'] = args.batch_size
    if args.dropout or args.dropout == 0.0:
        params['dropout'] = args.dropout
    if args.backbone:
        params['backbone'] = args.backbone
    if args.freeze or args.freeze == 0.0:
        params['freeze'] = args.freeze
    if args.colormap:
        params['colormap'] = args.colormap
    if args.augmentation_techniques:
        params['augmentation_techniques'] = args.augmentation_techniques
    if args.weighted_gradients:
        params['weighted_gradients'] = args.weighted_gradients
    if args.normalized_weights:
        params['normalized_weights'] = args.normalized_weights
    if args.regularized_loss:
        params['regularized_loss'] = args.regularized_loss
    if args.epochs:
        params['epochs'] = args.epochs
    if args.iterations:
        params['iterations'] = args.iterations

    """ -arguments: True if set as parameter """
    params['augmentation'] = args.augmentation

    return params

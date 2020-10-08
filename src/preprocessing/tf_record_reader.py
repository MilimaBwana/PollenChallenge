import tensorflow as tf
import numpy as np
import cv2 as cv


def read(mode, params):
    """
    @param mode: Either tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL or tf.estimator.ModeKeys.PREDICT.
    @param params: dictionary with important, predefined hyperparameters for the dataset. Must contain
            'dataset', 'batch_size', 'input_shape', 'augmentation' and 'colormap'.
    @return a parsed tfRecordDataset."""

    if mode is tf.estimator.ModeKeys.TRAIN:

        path_train_tf_record = params['dataset'].tf_record_train
        ds = tf.data.TFRecordDataset(path_train_tf_record)
        ds = ds.shuffle(5000)

    elif mode is tf.estimator.ModeKeys.EVAL:

        path_val_tf_record = params['dataset'].tf_record_val
        ds = tf.data.TFRecordDataset(path_val_tf_record)

    elif tf.estimator.ModeKeys.PREDICT:

        path_test_tf_record = params['dataset'].tf_record_test
        ds = tf.data.TFRecordDataset(path_test_tf_record)

    else:
        raise ValueError('No valid option')

    # Tests
    #for x in ds:
    #     __parse_example(x, params, mode)
    ds = ds.map(lambda x: __parse_example(x, params, mode), num_parallel_calls=8)
    if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
        ds = ds.batch(batch_size=params['batch_size'], drop_remainder=True)
    elif mode == tf.estimator.ModeKeys.PREDICT:
        ds = ds.batch(batch_size=1, drop_remainder=False)
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return ds


def __parse_example(serialized_example, params, mode):
    """Parses a image/label/filename-tuple.
    @param serialized_example: one serialized example out of a tf record.
    @param params: dictionary containing important hyperparameters. Must contain
            'batch_size', 'input_shape', 'augmentation' and 'colormap' as a key.
            If 'augmentation', the 'augmentation_techniques' key is also needed.
            If a model using transfer learning  is trained, 'backbone' key is necessary.
    @return a parsed (image, label, filename)-tuple.
    """
    context, sequence = tf.io.parse_single_sequence_example(
        serialized_example,
        context_features={
            'image_raw': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.int64),
            'filename': tf.io.FixedLenFeature([], tf.string)
        })

    # if params['input_shape'][2] = 1, then grayscale, if = 3, then RGB image
    img = tf.image.decode_png(context['image_raw'], channels=params['input_shape'][2])
    img = tf.cast(img, dtype=tf.float32)

    if params['augmentation'] and mode == tf.estimator.ModeKeys.TRAIN:
        img = __augment(img, params)

    if tf.shape(img)[0] <= params['input_shape'][0] and tf.shape(img)[1] <= params['input_shape'][1]:
        # Embed the image into a black background, if image is smaller than target size.
        offset_height = (params['input_shape'][0] - tf.shape(img)[0]) // 2
        offset_width = (params['input_shape'][1] - tf.shape(img)[1]) // 2
        img = tf.image.pad_to_bounding_box(img, offset_height=offset_height, offset_width=offset_width,
                                           target_height=params['input_shape'][0],
                                           target_width=params['input_shape'][1])
    else:
        # Resize while keeping the aspect ratio the same without distortion if image larger than target size.
        img = tf.image.resize_with_pad(img, target_height=params['input_shape'][0],
                                       target_width=params['input_shape'][1])

    if params['colormap'].lower() != 'none' and params['input_shape'][2] == 3:
        img = __apply_color_map(img, params)

    if 'backbone' in params:
        # If transfer learning is used, preprocesses according to used backbone.
        #TODO: To be edited if new backbone
        if params['backbone'].lower() == 'vgg16':
            img = tf.keras.applications.vgg16.preprocess_input(
                img, data_format="channels_last")
        elif params['backbone'].lower() == 'inception':
            img = tf.keras.applications.inception_v3.preprocess_input(
                img, data_format="channels_last")
        elif params['backbone'].lower() == 'resnet':
            img = tf.keras.applications.resnet.preprocess_input(
                img, data_format="channels_last")
        elif params['backbone'].lower() == 'resnetv2':
            img = tf.keras.applications.resnet_v2.preprocess_input(
                img, data_format="channels_last")
        elif params['backbone'].lower() == 'densenet':
            img = tf.keras.applications.densenet.preprocess_input(
                img, data_format="channels_last")
        else:
            img = (2 * img / 255) - 1 # normalize to [-1,1]

    else:
        img = img / 255  # normalize to [0,1]

    label = tf.cast(context['label'], tf.int64)
    filename = tf.cast(context['filename'], tf.string)
    return img, label, filename


def __filter_big_images(img, params):
    """Filters the image from the data set if the image size is larger than the target size.
    @param img: image
    @param params: dictionary, which must contain 'input_shape' as a key.
    @return: false, if image size is larger than target size, else true.
    """
    if tf.shape(img)[0] > params['input_shape'][0] or tf.shape(img)[1] > params['input_shape'][1]:
        return False
    else:
        return True


def __apply_color_map(img, params):
    """
    @param img: image
    @param params: dictionary, which must contain 'colormap'.
    @return: a colormapped image.
    """

    def __cv2_apply_color_map(img, colormap):
        if colormap == 'viridis':
            # cv2 only allows uint8 images for colormapping.
            img = cv.applyColorMap(np.uint8(img), cv.COLORMAP_VIRIDIS)

        return img
    if params['colormap'].lower() == 'viridis':
        img = tf.py_function(func=__cv2_apply_color_map, inp=[img, params['colormap']], Tout=tf.float32)
    return img


def __augment(img, params):
    """ Randomly augments the given image.
    @param img: image to augment.
    @param params. dictionary which must contain 'augmentation_techniques' as a key.
        Possible augmentations: flip_left_right, flip_up_down, crop, rotate, distort_color, noise.
        Value of key augmentation_techniques needs to be a list.
    @return augmented image. """
    augmentations = params['augmentation_techniques']

    augmentation_functions = {
        'flip_left_right': __flip_left_right,
        'flip_up_down': __flip_up_down,
        'crop': __crop,
        'rotate': __rotate,
        'noise': __noise,
        'distort_color': __distort_color}

    input_shape = params['input_shape']
    for augmentation in augmentations:
        # Probability of augmenting is 0.25 """
        if augmentation in augmentation_functions.keys():
            f = augmentation_functions[augmentation]
            img = tf.cond(tf.random.uniform([], 0, 1) >= 0.75, lambda: f(img, input_shape), lambda: img)
        else:
            raise ValueError('No valid augmentation: ', augmentation)

    return img


def __crop(img, input_shape):
    """Randomly crops the image on the sides. Crop on each dimension between 1 and 15 %.
    @param img: image to crop.
    @param input_shape: shape of the outputted image.
    @return the cropped image.
    """
    height = tf.cast(tf.shape(img)[0], tf.dtypes.float32)
    width = tf.cast(tf.shape(img)[1], tf.dtypes.float32)
    crop_height = tf.cast(height * tf.random.uniform(shape=[], minval=0.85, maxval=0.99, dtype=tf.float32),
                          dtype=tf.int32)
    crop_width = tf.cast(width * tf.random.uniform(shape=[], minval=0.85, maxval=0.99, dtype=tf.float32),
                         dtype=tf.int32)
    # if params['input_shape'][2] = 1, then grayscale, if = 3 then RGB image
    img = tf.image.random_crop(img, [crop_height, crop_width, input_shape[2]])
    return img


def __rotate(img, input_shape):
    """ Rotates the image randomly between -45 and 45 degrees and fills the borders with black pixels.
    @param img: image to rotate.
    @param input_shape: shape of the outputted image.
    @return: the rotated image.
    """

    def __cv2_rotate(image):
        """ Rotates the image by random degree. """
        num_rows, num_cols = image.shape[:2]
        deg = np.float64(np.random.uniform(-45, 45, size=1)) # For some reason deg must be float64, not float32
        rotation_matrix = cv.getRotationMatrix2D((num_cols / 2, num_rows / 2), deg, 1)
        image = cv.warpAffine(np.float32(image), rotation_matrix, (num_cols, num_rows))
        # In case of only one channel, warpAffine removes channel dimension.

        return image

    #deg = tf.random.uniform([], -45, 45, tf.dtypes.float64)
    img = tf.py_function(func=__cv2_rotate, inp=[img], Tout=tf.float32)
    if input_shape[2] == 1:
        # In case of only one channel, _cv2_rotate removes channel dimension, which needs to be added afterwards.
        img = tf.expand_dims(img, axis=-1)
    return img



def __flip_left_right(image, input_shape):
    """Flips image left/right.
    @param image: image to flip
    @param input_shape: shape of the outputted image.
    @return: the flipped image.
    """
    x = tf.image.flip_left_right(image)
    return x


def __flip_up_down(image, input_shape):
    """Flips image left/right.
    @param image: image to flip
    @param input_shape: shape of the outputted image.
    @return: the flipped image.
    """
    x = tf.image.flip_up_down(image)
    return x


def __noise(image, input_shape):
    """ Adds random noise from a normal distribution.
    @param image: image to put noise onto.
    @param input_shape: shape of the outputted image.
    @return: the noised image.
    """
    noise = tf.random.normal(shape=tf.shape(image), mean=15, stddev=1, dtype=tf.float32)
    image = tf.add(image, noise)
    image = tf.clip_by_value(image, 0, 255)
    return image


def __distort_color(image, input_shape):
    """Changes color randomly.
    @param image: image to put distort color. Needs to have 3 channels.
    @param input_shape: shape of the outputted image.
    @return: the color-distorted image."""
    x = tf.image.random_hue(image, max_delta=0.08)
    x = tf.image.random_saturation(x, lower=0.7, upper=1.3)
    x = tf.image.random_brightness(x, max_delta=0.05)
    x = tf.image.random_contrast(x, lower=0.7, upper=1.3)
    return x

import cv2 as cv
import numpy as np


def augment(img):
    """
    Augments the given image. Possible augmentations: flip_left_right, flip_up_down, crop, rotate, noise.
    @param img: image to augment.
    @return augmented image.
    """
    augmentation_functions = [__rotate, __flip_left_right, __flip_up_down, __crop, __noise]

    for f in augmentation_functions:
        """ Probability of augmenting is 0.75. """
        img = f(img) if np.random.uniform(low=0, high=1, size=None) >= 0.25 else img

    return img


def __rotate(image):
    """ Rotates the image by a random degree between -15 and 15 degrees.
    @param image: image to rotate
    @return: the rotated image."""
    degree = np.random.uniform(low=-45, high=45, size=None)
    num_rows, num_cols = image.shape[:2]

    rotation_matrix = cv.getRotationMatrix2D((num_cols / 2, num_rows / 2), degree, 1)
    image = cv.warpAffine(image, rotation_matrix, (num_cols, num_rows))
    return image


def __flip_left_right(image):
    """Flips image left/right.
    @param image: image to flip
    @return: the flipped image.
    """
    image = cv.flip(image, 1)
    return image


def __flip_up_down(image):
    """Flips image up/down.
    @param image: image to flip
    @return: the flipped image."""
    image = cv.flip(image, 0)
    return image


def __noise(image):
    """ Adds random gaussian noise to the image.
    @param image: image to put noise onto.
    @return the noised image"""
    image = image + np.random.randn(image.shape[0], image.shape[1]) * 15
    image = np.clip(image, a_min=0, a_max=255)
    return image


def __crop(image):
    """ Randomly crops images at the sides between 5 and 15 percent.
    @param image: image to crop.
    @return the cropped image.
    """
    left = int(image.shape[0] * np.random.uniform(low=0.05, high=0.15, size=None))  # Crop between 1 and 10 %
    right = int(image.shape[0] * np.random.uniform(low=0.85, high=0.95, size=None))
    top = int(image.shape[1] * np.random.uniform(low=0.05, high=0.15, size=None))
    bottom = int(image.shape[1] * np.random.uniform(low=0.85, high=0.95, size=None))
    x = image[top:bottom, left:right]
    return x

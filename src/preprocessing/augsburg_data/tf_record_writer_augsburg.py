import tensorflow as tf
import sys
import os
import csv
from pathlib import Path
from collections import defaultdict, Counter
import random
import numpy as np
import re

""" Syspath needs to include parent directory "pollen classification" and "Code" to find sibling 
modules and database."""
file_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))) + "/"
sys.path.append(file_path)

from config import config as cfg
from preprocessing import record_write_ops, pipeline_ops


def write(dataset, read_from_csv=False):
    """ Creates a training, validation and test tfRecord out of the png-Images in the given 'root_path'.
    In addition, a json-file with the number of samples per class is created for the training
    and validation dataset.py
    @param dataset: Instance of dataset_config.Dataset. Name is 'original_[15|31]', 'upsample_[15|31]' or
            'downsample_[15|31]'
    @param read_from_csv: if true, the train/val/test split is done according to the given csv files.
            Currently only available for 'original' dataset.
    """

    if cfg.UPSAMPLE_NAME in dataset.name.lower():
        __write_upsample(dataset, read_from_csv)
    elif cfg.DOWNSAMPLE_NAME in dataset.name.lower():
        __write_downsample(dataset, read_from_csv)
    elif cfg.ORIGINAL_NAME_NAME in dataset.name.lower():
        __write_original(dataset, read_from_csv)
    else:
        raise ValueError('No valid dataset')


def __write_original(dataset, read_from_csv=False):
    """
    Write tf records and json-files for the original dataset.
    @param dataset: Instance of dataset_config.Dataset
    @param read_from_csv: Splitting is done after shuffling the
    files according to cfg.TRAIN_SPLIT and cfg.VAL_SPLIT, if read_from_csv is False.
    Otherwise, splitting is done according to csv files.
    @return: Nothing
    """
    if read_from_csv:
        with open(cfg.CSV_TRAIN_LABELS_15, encoding="utf8") as csv_file:
            train_files = list(csv.reader(csv_file, delimiter=','))

        with open(cfg.CSV_VAL_LABELS_15, encoding="utf8") as csv_file:
            val_files = list(csv.reader(csv_file, delimiter=','))

        with open(cfg.CSV_TEST_LABELS_15, encoding="utf8") as csv_file:
            test_files = list(csv.reader(csv_file, delimiter=','))

        ''' String numerical label to int label'''
        train_files = [[str(dataset.data_dir) + '/' + x, int(y)] for x, y in train_files]
        val_files = [[str(dataset.data_dir) + '/' + x, int(y)] for x, y in val_files]
        test_files = [[str(dataset.data_dir) + '/' + x, int(y)] for x, y in test_files]

        random.shuffle(train_files)
        random.shuffle(val_files)
        random.shuffle(test_files)

    else:
        labeled_subdirectories = pipeline_ops.get_labeled_valid_subdirectories(dataset.name, dataset.data_dir,
                                                                               dictionary=dataset.dictionary,
                                                                               consider_misspelled=True)

        """ Create (filepath,label)-List with int-label for all png images """
        labeled_file_paths = pipeline_ops.get_labeled_png_paths(labeled_subdirectories)
        length_all_files = len(labeled_file_paths)

        assert (length_all_files > 0), "No images in given root path"

        """Split file names into train, val and test """
        # random.seed(42)
        random.shuffle(labeled_file_paths)
        number_train_samples = int(cfg.TRAIN_SPLIT * length_all_files)
        number_val_samples = int(cfg.VAL_SPLIT * length_all_files)

        train_files = labeled_file_paths[:number_train_samples]
        val_files = labeled_file_paths[number_train_samples:(number_train_samples + number_val_samples)]
        test_files = labeled_file_paths[(number_train_samples + number_val_samples):]

    record_write_ops.write_train_test_val_json(dataset, train_files, val_files, test_files)


def __write_upsample(dataset, read_from_csv=False):
    """
    Write tf records and json-files for the upsampled dataset.  Splitting is done after shuffling the
    files according to cfg.TRAIN_SPLIT and cfg.VAL_SPLIT.
    @param dataset: Instance of dataset_config.Dataset
    @param read_from_csv: can only be False here, otherwise error is thrown.
    @return: Nothing
    """
    if read_from_csv:
        raise NotImplementedError('No csv files for upsample dataset.')

    labeled_subdirectories = pipeline_ops.get_labeled_valid_subdirectories(dataset.name, dataset.data_dir,
                                                                           dictionary=dataset.dictionary,
                                                                           consider_misspelled=True)
    labeled_subdirectories_original = [x for x in labeled_subdirectories if cfg.UPSAMPLE_DIR_EXTENSION not in x[0]]
    labeled_subdirectories_upsample = [x for x in labeled_subdirectories if cfg.UPSAMPLE_DIR_EXTENSION in x[0]]

    """ Create (filepath,label)-List with int-label for all png images """
    labeled_file_paths_original = pipeline_ops.get_labeled_png_paths(labeled_subdirectories_original)
    labeled_file_paths_upsample = pipeline_ops.get_labeled_png_paths(labeled_subdirectories_upsample)
    length_all_files = len(labeled_file_paths_original) + len(labeled_file_paths_upsample)

    assert (length_all_files > 0), "No images in given root path"

    random.shuffle(labeled_file_paths_upsample)
    random.shuffle(labeled_file_paths_original)

    number_train_samples = int(cfg.TRAIN_SPLIT * length_all_files)
    number_val_samples = int(cfg.VAL_SPLIT * length_all_files)

    assert len(
        labeled_file_paths_upsample) < 0.5 * number_train_samples, "Training set must consist of 50 or more " \
                                                                   "percent original images."

    """Split file names into train, val and test. Upsampled images are only in the train set. """
    number_train_samples_original = int(number_train_samples - len(labeled_file_paths_upsample))

    train_files = labeled_file_paths_upsample + labeled_file_paths_original[
                                                :number_train_samples_original]
    random.shuffle(train_files)

    val_files = labeled_file_paths_original[
                number_train_samples_original: (number_train_samples_original + number_val_samples)]
    test_files = labeled_file_paths_original[(number_train_samples_original + number_val_samples):]

    record_write_ops.write_train_test_val_json(dataset, train_files, val_files, test_files)


def __write_downsample(dataset, read_from_csv=False, threshold=3):
    """
    Write tf records and json-files for a downsampled dataset. Train files are downsampled, val and test files
    stay at the same amount as in the orginal dataset.
    @param dataset: Instance of dataset_config.Dataset
    @param read_from_csv: can only be False here, otherwise error is thrown.
    @return: Nothing
    """
    if read_from_csv:
        raise NotImplementedError('No csv files for downsample dataset.')
    else:
        labeled_subdirectories = pipeline_ops.get_labeled_valid_subdirectories(dataset.name, dataset.data_dir,
                                                                               dictionary=dataset.dictionary,
                                                                               consider_misspelled=True)
        labeled_file_paths = pipeline_ops.get_labeled_png_paths(labeled_subdirectories)
        length_all_files = len(labeled_file_paths)

        assert (length_all_files > 0), "No images in given root path"

        """Split file names into train, val and test """
        random.shuffle(labeled_file_paths)
        number_train_samples = int(cfg.TRAIN_SPLIT * length_all_files)
        number_val_samples = int(cfg.VAL_SPLIT * length_all_files)

        train_files = labeled_file_paths[:number_train_samples]
        val_files = labeled_file_paths[number_train_samples:(number_train_samples + number_val_samples)]
        test_files = labeled_file_paths[(number_train_samples + number_val_samples):]
        counter_classes = Counter([c[1] for c in train_files])

        """ Threshold_samples = Number of samples in minority class times threshold. """
        threshold_samples = min(counter_classes.values()) * threshold
        train_files_downsample = []

        for key in counter_classes:

            if counter_classes[key] > threshold_samples:
                """ Downsample, if this class has more samples than 'threshold_samples'. """
                all_files_class = [f[0] for f in train_files if f[1] == key]
                chosen_files_class = np.random.choice(a=all_files_class,
                                                      size=threshold_samples)  # a must be 1-dimensional
                chosen_files_class = [[f, key] for f in chosen_files_class]
                train_files_downsample += chosen_files_class
            else:
                """ Keep all files, if this class has less samples than 'threshold_samples'."""
                all_files_class = [f for f in train_files if f[1] == key]
                train_files_downsample += all_files_class

        random.shuffle(train_files_downsample)

        record_write_ops.write_train_test_val_json(dataset, train_files, val_files, test_files)

from pathlib import Path
import re
from preprocessing.italian_data import preprocessing_ops_italian as preprocessing_italian
from preprocessing.augsburg_data import preprocessing_ops_augsburg as preprocessing_augsburg

"""
File which implements facade methods for preprocessing.
"""


def get_labeled_valid_subdirectories(dataset_name, data_dir, dictionary, consider_misspelled=True,
                                     mask='object'):
    """
    Calls the corresponding function 'get_labeled_valid_subdirectories' for a 'dataset'.
    @param dataset_name: Name of the dataset, also containing number of classes after '_'
    @param data_dir: root directory for

    """
    # TODO: Editing, when new dataset.py
    if re.match('[a-zA-Z]+_4$', dataset_name):
        return preprocessing_italian.get_labeled_valid_subdirectories(dataset_name, data_dir, mask)
    elif re.match('[a-zA-Z]+_15$', dataset_name) or re.match('[a-zA-Z]+_31$', dataset_name):
        return preprocessing_augsburg.get_labeled_valid_subdirectories(dataset_name, data_dir, dictionary,
                                                                       consider_misspelled)
    else:
        raise ValueError('No valid dataset name.')


def get_valid_subdirectories(dataset_name, data_dir, dictionary, consider_misspelled=True,
                             mask='object'):
    """
    Calls the corresponding function 'get_valid_subdirectories' for a 'dataset'.
    """
    # TODO: Editing, when new dataset.py
    if re.match('[a-zA-Z]+_4$', dataset_name):
        return preprocessing_italian.get_valid_subdirectories(dataset_name, data_dir, mask)
    elif re.match('[a-zA-Z]+_15$', dataset_name) or re.match('[a-zA-Z]+_31$', dataset_name):
        return preprocessing_augsburg.get_valid_subdirectories(dataset_name, data_dir, dictionary, consider_misspelled)
    else:
        raise ValueError('No valid dataset name.')


def get_labeled_png_paths(labeled_subdirectories):
    """
    @param labeled_subdirectories: list of [subdirectory,label] with a string representing the subdirectory and an int label.
    @return: Returns [filepath,label]-List with int-label for all png images in labeled_subdirectories.
    """
    labeled_file_paths = [[f, label] for subdirectory, label in labeled_subdirectories for f in
                          Path(subdirectory).glob('*.png') if f.is_file()]
    return labeled_file_paths


def get_png_paths(subdirectories):
    """
    @param subdirectories: list of strings representing the subdirectory.
    @return: filepath-List with all png images in all subdirectories.
    """
    file_paths = [f for subdirectory in subdirectories for f in Path(subdirectory).glob('*.png') if f.is_file()]
    return file_paths

import os
from pathlib import Path
from config import config as cfg


def get_valid_subdirectories(dataset_name, data_dir, dictionary, consider_misspelled=True):
    """
    Get all subdirectories within root_path if the subdirectory is in the dictionary corresponding to the dataset.py.
    @param dataset_name: name of the dataset.
    @param data_dir = parent directory containing all data.
    @param dictionary = mapping between class names(keys) and labels(values)
    @param consider_misspelled: if true, misspelled folders are taken into account. See cfg.POLLEN_DIR_MATCHER
    @return valid_subdirectories: a list of the valid subdirectories.
    """
    valid_subdirectories = []
    subdirectories = [x[0] for x in os.walk(Path(data_dir))]

    """ Look for subdirectories with relevant classes. """
    for subdirectory in subdirectories:
        folder = os.path.split(subdirectory)[1]
        if folder in cfg.POLLEN_DIR_MATCHER and consider_misspelled:
            """ Misspelled folder. """
            label = cfg.POLLEN_DIR_MATCHER[folder]
        elif cfg.UPSAMPLE_DIR_EXTENSION in folder and cfg.UPSAMPLE_NAME in dataset_name:
            """ folder with upsampled data. """
            label = folder.split('_')[0]
        else:
            """ Correct spelled folder. """
            label = folder

        if label in dictionary:
            valid_subdirectories.append((str(subdirectory)))

    return valid_subdirectories


def get_labeled_valid_subdirectories(dataset_name, data_dir, dictionary, consider_misspelled=True):
    """
    Get all subdirectories within data_dir if the subdirectory is a key in the dictionary and labels them.
    @param dataset_name: name of the dataset.
    @param data_dir: parent directory containing all data
    @param dictionary: mapping between class names(keys) and labels(values)
    @param consider_misspelled: if true, misspelled folders are taken into account.
    @return valid_subdirectories: a list of tuples (valid_subdirectory, corresponding class label (int)).
    """
    labeled_subdirectories = []
    subdirectories = [x[0] for x in os.walk(Path(data_dir))]

    for subdirectory in subdirectories:
        folder = os.path.split(subdirectory)[1]
        if folder in cfg.POLLEN_DIR_MATCHER and consider_misspelled:
            """ misspelled folder """
            label = cfg.POLLEN_DIR_MATCHER[folder]
        elif cfg.UPSAMPLE_DIR_EXTENSION in folder and cfg.UPSAMPLE_NAME in dataset_name:
            """ folder with upsampled data. """
            label = folder.split('_')[0]
        else:
            """ correct spelled folder """
            label = folder

        if label in dictionary:
            labeled_subdirectories.append((str(subdirectory), dictionary[label]))

    return labeled_subdirectories



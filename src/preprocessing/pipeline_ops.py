import os
from pathlib import Path
from collections import defaultdict
import json
import sys

from config import config as cfg


def get_valid_subdirectories(dataset_name, data_dir, mask='object'):
    """
    Get all subdirectories within dataset.data_dir if the subdirectory is in the dictionary cfg.POLLEN_DIR_MATCHER.
    If cfg.UPSAMPLE_NAME is in 'dataset_name', upsampled directories are also valid.
    @param dataset_name: name of the dataset.
    @param data_dir = parent directory containing all data.
    @param mask: folder extension.
    @return a list of the valid subdirectories.
    """

    valid_subdirectories = []
    # Get all subdirectories, which dont contain a subdirectory themselves.
    subdirectories = [x[0] for x in os.walk(Path(data_dir)) if not x[1]]

    # Get subdirectories with correct mask extension
    mask_extension = get_dir_mask_extension(mask)
    subdirectories = [x for x in subdirectories if mask_extension in x]

    for subdirectory in subdirectories:
        if cfg.UPSAMPLE_DIR_EXTENSION in subdirectory and cfg.UPSAMPLE_NAME in dataset_name:
            # folder with upsampled data.
            valid_subdirectories.append(subdirectory)
        elif cfg.UPSAMPLE_DIR_EXTENSION not in subdirectory:
            valid_subdirectories.append(str(subdirectory))

    return valid_subdirectories


def get_labeled_valid_subdirectories(dataset_name, data_dir, mask='object'):
    """ Get all subdirectories within data_dir if the subdirectory is in the dictionary cfg.POLLEN_DIR_MATCHER
    and labels them. If cfg.UPSAMPLE_NAME is in 'dataset_name', upsampled directories are also valid.
    @param dataset_name: name of the dataset.
    @param data_dir: parent directory containing all data
    @param mask: folder extension.
    @return valid_subdirectories: a list of tuples (valid_subdirectory, corresponding class label (int)).
    """
    labeled_subdirectories = []
    # Get all subdirectories, which dont contain a subdirectory themselves.
    subdirectories = [x[0] for x in os.walk(Path(data_dir)) if not x[1]]

    # Get subdirectories with correct mask extension
    mask_extension = get_dir_mask_extension(mask)
    subdirectories = [x for x in subdirectories if mask_extension in x]

    for subdirectory in subdirectories:
        folder = os.path.split(os.path.split(subdirectory)[0])[1]
        if cfg.UPSAMPLE_DIR_EXTENSION in subdirectory and cfg.UPSAMPLE_NAME in dataset_name:
            # folder with upsampled data.
            label = int(folder) - 1
            labeled_subdirectories.append((str(subdirectory), label))
        elif cfg.UPSAMPLE_DIR_EXTENSION not in subdirectory:
            label = int(folder) - 1
            labeled_subdirectories.append((str(subdirectory), label))

    return labeled_subdirectories


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


'''def label_test_set(dataset, directory):
    """
    Labels each file in the test dataset
    @param dataset: config.DatasetConfig object
    @param directory:  Directory in which the unlabeled test files are located
    @return:
    """

    file_list = pipeline_ops.get_png_paths(directory)
    if dataset.test_label_json:
        try:
            # Create (filepath,label)-List with int-label for all png images in test dataset using json file
            with open(dataset.test_label_json, 'r') as file:
                raw_dict = json.load(file)
                filename_label_dict = defaultdict()
                for entry in raw_dict:
                    filename_label_dict[entry['Filename']] = int(entry['Class']) - 1  # class shift
                test_files = [[x, filename_label_dict[os.path.basename(x)]] for x in file_list]

        except:
            test_files = [[x, 0] for x in file_list]  # Add artificial label 0 to all test files
    else:
        test_files = [[x, 0] for x in file_list]  # Add artificial label 0 to all test files

    return test_files'''


def get_dir_mask_extension(mask='object'):
    if mask == 'object':
        extension = '_OBJ'
    elif mask == 'mask':
        extension = '_MASK'
    elif mask == 'segmentation':
        extension = '_SEGM'
    else:
        extension = ''

    return extension

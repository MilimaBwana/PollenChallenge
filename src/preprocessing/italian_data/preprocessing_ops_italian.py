import os
from pathlib import Path
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
    """ Get all subdirectories, which dont contain a subdirectory themselves."""
    subdirectories = [x[0] for x in os.walk(Path(data_dir)) if not x[1]]

    """Get subdirectories with correct mask extension"""
    mask_extension = __get_dir_mask_extension(mask)
    subdirectories = [x for x in subdirectories if mask_extension in x]

    for subdirectory in subdirectories:
        if cfg.UPSAMPLE_DIR_EXTENSION in subdirectory and cfg.UPSAMPLE_NAME in dataset_name:
            """ folder with upsampled data. """
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
    """ Get all subdirectories, which dont contain a subdirectory themselves."""
    subdirectories = [x[0] for x in os.walk(Path(data_dir)) if not x[1]]

    """Get subdirectories with correct mask extension"""
    mask_extension = __get_dir_mask_extension(mask)
    subdirectories = [x for x in subdirectories if mask_extension in x]

    for subdirectory in subdirectories:
        folder = os.path.split(os.path.split(subdirectory)[0])[1]
        if cfg.UPSAMPLE_DIR_EXTENSION in subdirectory and cfg.UPSAMPLE_NAME in dataset_name:
            """ folder with upsampled data. """
            label = int(folder) - 1
            labeled_subdirectories.append((str(subdirectory), label))
        elif cfg.UPSAMPLE_DIR_EXTENSION not in subdirectory:
            label = int(folder) - 1
            labeled_subdirectories.append((str(subdirectory), label))

    return labeled_subdirectories


def __get_dir_mask_extension(mask='object'):
    if mask == 'object':
        extension = '_OBJ'
    elif mask == 'mask':
        extension = '_MASK'
    elif mask == 'segmentation':
        extension = '_SEGM'
    else:
        extension = ''

    return extension

import sys
import os
import random
import argparse

""" Syspath needs to include parent directory "pollen_classification" to find sibling modules ."""
file_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + "/"
sys.path.append(file_path)


from config import config as cfg
from preprocessing import record_write_ops, pipeline_ops
from config import dataset_config


def main():
    parser = argparse.ArgumentParser(description='Tf Record Writer Parameter.')
    parser.add_argument('--dataset_name', type=str,
                        help='Used Dataset. Original_4 or upsample_4')
    args = parser.parse_args()

    dataset_name = 'original_4'

    if args.dataset_name:
        dataset_name = args.dataset_name.lower()

    write(dataset_name)


def write(dataset_name, mask='object'):
    dataset = dataset_config.Dataset(dataset_name)
    if cfg.UPSAMPLE_NAME in dataset.name.lower():
        __write_upsample(dataset, mask=mask)
    elif cfg.ORIGINAL_NAME in dataset.name.lower():
        __write_original(dataset, mask=mask)
    else:
        raise ValueError('No valid dataset')


def __write_original(dataset, mask='object'):
    """
    Write tf records and json-files for the original dataset. Files in train directory are splitted into train
    and val files. Test set is an extra directory.
    @param dataset: Dataset object, i.a. containing number of classes
    @param mask: images used in italian dataset. Can be 'object', 'mask' or 'segmentation'. Default 'object'.
    @return: Nothing
    """

    labeled_subdirectories_trainval = pipeline_ops.get_labeled_valid_subdirectories(dataset.name,
                                                                                    os.path.join(dataset.data_dir,
                                                                                                 'train'),
                                                                                    mask=mask)
    labeled_subdirectories_test = pipeline_ops.get_labeled_valid_subdirectories(dataset.name,
                                                                        os.path.join(dataset.data_dir, 'test'),
                                                                        mask='None')

    # Create (filepath,label)-List with int-label for all png images
    labeled_file_paths_trainval = pipeline_ops.get_labeled_png_paths(labeled_subdirectories_trainval)
    length_all_files = len(labeled_file_paths_trainval)
    test_files = pipeline_ops.get_labeled_png_paths(labeled_subdirectories_test)

    # Split file names into train and val, test has own directory
    random.shuffle(labeled_file_paths_trainval)
    number_train_samples = int((cfg.TRAIN_SPLIT / (cfg.TRAIN_SPLIT + cfg.VAL_SPLIT)) * length_all_files)

    train_files = labeled_file_paths_trainval[:number_train_samples]
    val_files = labeled_file_paths_trainval[number_train_samples:]

    record_write_ops.write_train_test_val_json(dataset, train_files, val_files, test_files)


def __write_upsample(dataset, mask='object'):
    """
    Write tf records and json-files for the upsampled dataset. Files in train directory are splitted into train
    and val files. Test set is an extra directory.
    @param dataset: Dataset object, i.a. containing number of classes
    @param mask: images used in italian dataset. Can be 'object', 'mask' or 'segmentation'. Default 'object'.
    @return: Nothing
    """

    labeled_subdirectories_trainval = pipeline_ops.get_labeled_valid_subdirectories(dataset.name,
                                                                                    os.path.join(dataset.data_dir,
                                                                                                 'train'),
                                                                                    mask=mask)
    labeled_subdirectories_test = pipeline_ops.get_labeled_valid_subdirectories(dataset.name,
                                                                        os.path.join(dataset.data_dir, 'test'),
                                                                        mask='None')
    labeled_subdirectories_original = [x for x in labeled_subdirectories_trainval if
                                       cfg.UPSAMPLE_DIR_EXTENSION not in x[0]]
    labeled_subdirectories_upsample = [x for x in labeled_subdirectories_trainval if
                                       cfg.UPSAMPLE_DIR_EXTENSION in x[0]]

    # Create (filepath,label)-List with int-label for all png images
    labeled_file_paths_original = pipeline_ops.get_labeled_png_paths(labeled_subdirectories_original)
    labeled_file_paths_upsample = pipeline_ops.get_labeled_png_paths(labeled_subdirectories_upsample)
    test_files = pipeline_ops.get_labeled_png_paths(labeled_subdirectories_test)
    length_all_files = len(labeled_file_paths_original) + len(labeled_file_paths_upsample)

    assert (length_all_files > 0), "No images in given root path"

    random.shuffle(labeled_file_paths_upsample)
    random.shuffle(labeled_file_paths_original)

    number_train_samples = int((cfg.TRAIN_SPLIT / (cfg.TRAIN_SPLIT + cfg.VAL_SPLIT)) * length_all_files)

    assert len(
        labeled_file_paths_upsample) < 0.5 * number_train_samples, "Training set must consist of 50 or more " \
                                                                   "percent original images."

    # Split file names into train, val and test. Upsampled images are only in the train set.
    number_train_samples_original = int(number_train_samples - len(labeled_file_paths_upsample))
    train_files = labeled_file_paths_upsample + labeled_file_paths_original[
                                                :number_train_samples_original]
    random.shuffle(train_files)
    val_files = labeled_file_paths_original[number_train_samples_original:]

    record_write_ops.write_train_test_val_json(dataset, train_files, val_files, test_files)


if __name__ == "__main__":
    main()
from pathlib import Path
import sys
import os
import numpy as np
import cv2 as cv
import shutil
import argparse
from collections import Counter

""" Syspath needs to include parent directory "pollen_classification" to find sibling modules ."""

file_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + "/"
sys.path.append(file_path)

from config import config as cfg
from config import dataset_config
from preprocessing import pipeline_ops, cv2_augmentation


def main():
    parser = argparse.ArgumentParser(description='Dataset upsample parameter.')

    parser.add_argument('--max_fraction', type=float,
                        help='Determines the maximum of upsampled images as a fraction of the number of samples '
                             'in the majority class.')
    parser.add_argument('--max_multiplication', type=float,
                        help='Determines maximum m of upsampled images for each class. The threshold equals m times'
                             'number of samples in the respective class.')
    parser.add_argument('--dataset_name', type=str,
                        help='Used Dataset. Upsample_4 or original_4')

    args = parser.parse_args()
    max_fraction = 0.1
    max_multiplication = 5
    dataset_name = 'original_4'

    if args.max_fraction:
        max_fraction = args.max_fraction
    if args.max_multiplication:
        max_multiplication = args.max_multiplication
    if args.dataset_name:
        dataset_name = args.dataset_name.lower()

    dataset = dataset_config.Dataset(dataset_name)
    delete_and_upsample(dataset, max_fraction, max_multiplication)


def delete_and_upsample(dataset, max_fraction, max_multiplication):
    delete_upsampled_data(dataset)
    upsample_data(dataset, max_fraction=max_fraction, max_multiplication=max_multiplication)


def upsample_data(dataset, max_fraction=None, max_multiplication=None):
    """
    Creates artificial data to balance the data set regarding class distribution. This is done by adding
    augmented images of minority classes to the data set. The number of added images is determined by the minimum
    of two thresholds, which are calculated from 'max_fraction' and 'max_multiplication'.
    @param dataset: Instance of dataset_config.Dataset
    @param max_fraction: threshold, which determines number of upsampled images as a fraction of the number of samples
            in the majority class. Should be in (0,1).
    @param max_multiplication: Determines number of upsampled image per class, by number of samples
        in this class times max_multiplication.
    """
    assert (
            max_fraction is not None or max_multiplication is not None), 'Either max_fraction or max_multiplication must ' \
                                                                         'be not None'
    max_fraction = 1 if max_fraction is None else max_fraction
    max_multiplication = np.Inf if max_multiplication is None else max_multiplication

    labeled_subdirectories = pipeline_ops.get_labeled_valid_subdirectories(dataset.name, dataset.data_dir,
                                                                           dataset.reverse_dictionary,
                                                                           consider_misspelled=True, mask='object')
    labeled_file_paths = pipeline_ops.get_labeled_png_paths(labeled_subdirectories)
    counter_classes = Counter([c[1] for c in labeled_file_paths])

    # threshold_samples_1 = threshold  * max_number_of_files in one class.
    threshold_samples = int(max(counter_classes.values()) * max_fraction)

    for key in counter_classes:

        if counter_classes[key] < threshold_samples:
            # Upsample data. Number of images to upsample is the minimum of the 2 thresholds.
            number_upsampling = int(min(threshold_samples - counter_classes[key],
                                          counter_classes[key] * max_multiplication))
            print('Upsample: ' + str(dataset.dictionary[key]) + ' ' + str(number_upsampling))

            base_directories = [path[0] for path in labeled_subdirectories if key == path[1]]
            save_directory = [x for x in base_directories if os.path.basename(x) in dataset.reverse_dictionary][
                                 0] + cfg.UPSAMPLE_DIR_EXTENSION
            Path(save_directory).mkdir(parents=True, exist_ok=True)

            all_files_class = [f[0] for f in labeled_file_paths if f[1] == key]
            upsample_files = np.random.choice(a=all_files_class,
                                              size=number_upsampling)  # a must be 1-dimensional

            for idx, file in enumerate(upsample_files):
                img = cv.imread(str(file), 0)
                img = cv2_augmentation.augment(img)
                save_path = os.path.join(save_directory,
                                         os.path.splitext(os.path.basename(file))[0] + "_" + str(idx) + ".png")
                cv.imwrite(save_path, img)


def delete_upsampled_data(dataset):
    """
    Deletes the content of each directory with the extension 'cfg.UPSAMPLE_DIR_EXTENSION' and the directory itself.
    @param dataset: Instance of dataset_config.Dataset
    @return Nothing
    """
    upsample_data_dirs = [x[0] for x in os.walk(dataset.data_dir) if
                            os.path.isdir(x[0]) and cfg.UPSAMPLE_DIR_EXTENSION in x[0]]
    for directory in upsample_data_dirs:
        print('Deleting: ', directory)
        print('Deleting: ', directory)
        shutil.rmtree(directory, ignore_errors=True)


if __name__ == "__main__":
    main()

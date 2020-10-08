import sys
import os
import json
from pathlib import Path

""" Syspath needs to include parent directory "pollen classification" and "src" to find sibling 
modules and database."""
file_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))) + "/"
sys.path.append(file_path)

from config import dataset_config
from preprocessing.italian_data import preprocessing_ops_italian


def move_test_data(mask='object'):
    """
    Move files into correct subdirectory, corresponding to class label
    @param mask: currently  only object is allowed.
    @return: Nothing
    """
    dataset = dataset_config.Dataset('original_4')

    if dataset.test_label_json:
        with open(dataset.test_label_json, 'r') as file:
            raw_dict = json.load(file)
            for entry in raw_dict:
                source_filepath = dataset.data_dir + '/test/images/' + entry['Filename']
                target_dir = dataset.data_dir + '/test/images/' + entry[
                    'Class'] + '/test_' + preprocessing_ops_italian.get_dir_mask_extension(mask)
                Path(target_dir).mkdir(parents=True, exist_ok=True)
                target_filepath = target_dir + '/' + entry['Filename']
                os.rename(source_filepath, target_filepath)

if __name__ == "__main__":
    move_test_data()

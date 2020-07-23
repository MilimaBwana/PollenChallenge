import sys
import os
import argparse
import re

""" Syspath needs to include parent directory "pollen classification" and "Code" to find sibling 
modules and database."""
file_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + "/"
sys.path.append(file_path)

from config import dataset_config
from preprocessing.augsburg_data import tf_record_writer_augsburg as augsburg_writer
from preprocessing.italian_data import tf_record_writer_italian as italian_writer


def main():
    parser = argparse.ArgumentParser(description='Tf Record Writer Parameter.')
    parser.add_argument('--dataset_name', type=str,
                        help='Used Dataset. Original_[4|15|30] or upsample_[4|15|30] or downsample_[15|30]')
    parser.add_argument('-read_csv', action='store_true', help='Test/val split according to csv files.')
    args = parser.parse_args()

    dataset_name = 'upsample_4'

    if args.dataset_name:
        dataset_name = args.dataset_name.lower()

    read_csv = args.read_csv
    write(dataset_name, read_csv)


def write(dataset_name, read_csv=False):
    """
    Calls the corresponding tfRecordWriter for a dataset_name. A different tfRecordWriter is specified for each data
    record because of the different structure within the data. Acts as a facade for the different implementations.
    @param dataset_name: Name of the dataset, also containing number of classes after '_'
    @param read_csv: if true, train/val/test-split is done according to csv-file. Only available for dataset_name
        'original_4'.
    @return: Nothing
    """
    print('Start writing tf_records ...')

    dataset = dataset_config.Dataset(dataset_name)
    # TODO: To be edited if new dataset
    if re.match('[a-zA-Z]+_4$', dataset.name):
        italian_writer.write(dataset=dataset, mask='object')
    elif re.match('[a-zA-Z]+_15$', dataset.name):
        augsburg_writer.write(dataset=dataset, read_from_csv=read_csv)
    elif re.match('[a-zA-Z]+_31$', dataset.name):
        augsburg_writer.write(dataset=dataset)
    else:
        raise ValueError('No valid name: Number after _ must be 4, 30 or 15.')
    print('Finish writing tf_records ...')


if __name__ == "__main__":
    main()

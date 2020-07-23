import os
import sys
import json
import tensorflow as tf
import shutil


def file_len(f_name):
    """
    @param: filepath to txt-file
    @return: number of lines in text-file- """
    i = -1
    with open(f_name, encoding="utf-8") as f:
        for i, l in enumerate(f):
            pass
    return i + 1


def clear_directory(folder, clear_subdirectories=False):
    """ Deletes every file in the given folder.
    @param folder: given folder to delete content
    @param: clear_subdirectories: if true, content of subdirectories is also cleared.
    @return: Nothing.
    """
    if os.path.exists(folder):
        for the_file in os.listdir(folder):
            file_path = os.path.join(folder, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path) and clear_subdirectories:
                    shutil.rmtree(file_path)
            except Exception as e:
                print(e)


def multiply(iterable):
    """Multiplies each element of an iterable.
    @return: the product of every element. """
    product = 1
    for num in iterable:
        product = product * num
    return product


def is_debugging():
    """@return: Returns True, if script is debugging, False if in run mode."""
    debugging = True
    gettrace = getattr(sys, 'gettrace', None)
    if gettrace is None:
        pass
    elif gettrace():
        """ Debug mode, hide all GPU Devices and disable @tf.function. """
        #os.environ['CUDA_VISIBLE_ DEVICES'] = ''
        tf.config.experimental_run_functions_eagerly(True)

        print('Debugging')
    else:
        # Run mode
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        debugging = False
        print('Running')

    return debugging


def count_samples(json_dict):
    """Sums up the values in the 'json_dict' and returns the sum.
    @param json_dict: must be a dictionary saved by json.dump.
    @return: sum of the values.
    """
    with open(json_dict, 'r') as file:
        dict = json.load(file)
    count = 0

    for key, value in dict.items():
        count += value

    return count


def shift_labels_in_json(path_to_json, shift=2):
    with open(path_to_json, 'r') as file:
        list_of_dicts = json.load(file)

    shifted_list = []
    for dict_entry in list_of_dicts:
        tmp_dict = {'Filename': dict_entry['Filename'], 'Class': str(int(dict_entry['Class']) + shift)}
        shifted_list.append(tmp_dict)

    save_path = os.path.join(os.path.splitext(path_to_json)[0] + '_shifted.json')

    with open(save_path, 'w') as file:
        json.dump(shifted_list, file)











import tensorflow as tf
from pathlib import Path
from collections import defaultdict
import json


def __bytes_feature(value):
    """Returns a bytes_list from a string/byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def __int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint. Used for label."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def __serialize_example(filepath, label):
    """Serializes a (img_filepath, label)-tuple. The raw image and the filename are saved as a byte feature, the
     label as a int64-feature.
     @return: the serialized example."""
    img_bytes = open(filepath, 'rb').read()
    filepath_bytes = str(Path(*filepath.parts[-2:])).encode('utf-8')
    feature = {
        'image_raw': __bytes_feature(img_bytes),
        'label': __int64_feature(label),
        'filename': __bytes_feature(filepath_bytes)
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))


def __write_tf_record(files, savepath, count_classes_dict):
    """
    Serializes all 'files' to a tfRecord specified by 'path_tf_record'.
    See: https://www.tensorflow.org/tutorials/load_data/tfrecord
    @param files: List of (filepath,label)-pair. filepath is a string. Label is an int representing the class.
    @param savepath: A string representing the file name to save the tf record to.
    @param count_classes_dict: dictionary with class frequencies.
    @return Updated dictionary with class frequencies.
    """

    with tf.io.TFRecordWriter(savepath) as writer:
        for filepath, label in files:
            count_classes_dict[label] += 1
            example = __serialize_example(filepath, label)
            # Write serialized example to TFRecord file
            writer.write(example.SerializeToString())

    return count_classes_dict


def __write_json(savepath, dictionary):
    """ Saves a dictionary to 'savepath' as a json file."""
    with open(savepath, "w") as file:
        json.dump(dictionary, file)


def write_train_test_val_json(dataset, train_files, val_files, test_files):
    """ Writes train_files, test_files and val_files to a tf record.
    Create count classes dict with frequency per class. """
    count_classes_dict = defaultdict()

    for x in dataset.dictionary:
        count_classes_dict[x] = 0

    count_classes_dict = __write_tf_record(train_files, dataset.tf_record_train,
                                           count_classes_dict)
    count_classes_dict = __write_tf_record(val_files, dataset.tf_record_val,
                                           count_classes_dict)
    count_classes_dict = __write_tf_record(test_files, dataset.tf_record_test,
                                           count_classes_dict)
    __write_json(dataset.count_json, count_classes_dict)


def write_train_test_val_json_without_test(dataset, train_files, val_files, test_files):
    """ Writes train_files, test_files and val_files to a tf record.
    Create count classes dict with frequency per class. Test files are not counted. This is used for a test
    set without known labels."""
    count_classes_dict = defaultdict()

    for x in dataset.dictionary:
        count_classes_dict[x] = 0

    count_classes_dict = __write_tf_record(train_files, dataset.tf_record_train,
                                           count_classes_dict)
    count_classes_dict = __write_tf_record(val_files, dataset.tf_record_val,
                                           count_classes_dict)
    __write_json(dataset.count_json, count_classes_dict)
    __write_tf_record(test_files, dataset.tf_record_test, count_classes_dict)



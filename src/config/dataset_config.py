from config import config as cfg
from config.util_ops import create_dict_and_reverse_dict


class Dataset:

    def __init__(self, name):
        """
        Maps important config variables to the given 'name' of a dataset.
        Properties:
            name: name of dataset, must contain number of classes after '_'
            data_dir: directory in which dataset is stored.
            tf_record_train: path of tf record containing train samples.
            tf_record_val: path of tf record containing eval samples.
            tf_record_test: path of tf record containing test samples.
            count_json: json file with frequencies per class.
            dictionary: mapping between labels(keys) and pollen names (values).
            reverse_dictionary: mapping between pollen names (keys) and labels(values).
            num_bottom_classes: number of classes with a few samples.
        """

        self.__ds_name = name
        self.__num_classes = int(name.split('_')[1])
        self.__test_label_json = None
        num_classes = str(self.__num_classes)
        # TODO: to be edited, if new dataset
        if name == 'original_4':
            self.__data_dir = cfg.DATA_DIR_4
            self.__tf_record_train = self.__data_dir + '/' + cfg.ORIGINAL_NAME + '_' + num_classes + '_traindata.tfrecords'
            self.__tf_record_val = self.__data_dir + '/' + cfg.ORIGINAL_NAME + '_' + num_classes + '_valdata.tfrecords'
            self.__tf_record_test = self.__data_dir + '/' + cfg.ORIGINAL_NAME + '_' + num_classes + '_testdata.tfrecords'
            self.__count_json = self.__data_dir + '/' + cfg.ORIGINAL_NAME + '_' + num_classes + '_count_classes.json'
            self.__name_classes = self.__data_dir + '/original_4.names'
            self.__dictionary, self.__reverse_dictionary = create_dict_and_reverse_dict(self.__name_classes)
            self.__test_label_json = self.__data_dir + '/test_labels.json'
            self.__num_bottom_classes = 3
        elif name == 'upsample_4':
            self.__data_dir = cfg.DATA_DIR_4
            self.__tf_record_train = self.__data_dir + '/' + cfg.UPSAMPLE_NAME + '_' + num_classes + '_traindata.tfrecords'
            self.__tf_record_val = self.__data_dir + '/' + cfg.UPSAMPLE_NAME + '_' + num_classes + '_valdata.tfrecords'
            self.__tf_record_test = self.__data_dir + '/' + cfg.UPSAMPLE_NAME + '_' + num_classes + '_testdata.tfrecords'
            self.__count_json = self.__data_dir + '/' + cfg.UPSAMPLE_NAME + '_' + num_classes + '_count_classes.json'
            self.__name_classes = self.__data_dir + '/original_4.names'
            self.__dictionary, self.__reverse_dictionary = create_dict_and_reverse_dict(self.__name_classes)
            self.__test_label_json = self.__data_dir + '/test_labels.json'
            self.__num_bottom_classes = 3
        else:
            raise ValueError('No valid dataset')

    # Only allow read on variables

    def __get_name(self):
        return self.__ds_name

    def __get_num_classes(self):
        return self.__num_classes

    def __get_data_dir(self):
        return self.__data_dir

    def __get_tf_record_train(self):
        return self.__tf_record_train

    def __get_tf_record_val(self):
        return self.__tf_record_val

    def __get_tf_record_test(self):
        return self.__tf_record_test

    def __get_count_json(self):
        return self.__count_json

    def __get_name_classes(self):
        return self.__name_classes

    def __get_dictionary(self):
        return self.__dictionary

    def __get_reverse_dictionary(self):
        return self.__reverse_dictionary

    def __get_test_label_json(self):
        return self.__test_label_json

    def __get_num_bottom_classes(self):
        return self.__num_bottom_classes

    name = property(__get_name)
    num_classes = property(__get_num_classes)
    data_dir = property(__get_data_dir)
    tf_record_train = property(__get_tf_record_train)
    tf_record_val = property(__get_tf_record_val)
    tf_record_test = property(__get_tf_record_test)
    count_json = property(__get_count_json)
    name_classes = property(__get_name_classes)
    dictionary = property(__get_dictionary)
    reverse_dictionary = property(__get_reverse_dictionary)
    num_bottom_classes = property(__get_num_bottom_classes)
    test_label_json = property(__get_test_label_json)

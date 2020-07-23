from config import config as cfg


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
            dictionary: mapping between pollen names (keys) and labels(values).
            reverse_dictionary: mapping between labels(keys) and pollen names (values).
            num_bottom_classes: number of classes with a few samples.
        """

        self.__ds_name = name
        self.__num_classes = int(name.split('_')[1])
        #TODO: to be edited, if new dataset
        if name == 'original_15':
            self.__data_dir = cfg.DATA_DIR_15
            self.__tf_record_train = cfg.TFRECORD_TRAIN_ORIGINAL_15
            self.__tf_record_val = cfg.TFRECORD_VAL_ORIGINAL_15
            self.__tf_record_test = cfg.TFRECORD_TEST_ORIGINAL_15
            self.__count_json = cfg.COUNT_JSON_ORIGINAL_15
            self.__dictionary = cfg.DICT_CLASSES_15
            self.__reverse_dictionary = cfg.REVERSE_DICT_CLASSES_15
            self.__num_bottom_classes = 11
        elif name == 'upsample_15':
            self.__data_dir = cfg.DATA_DIR_15
            self.__tf_record_train = cfg.TFRECORD_TRAIN_UPSAMPLE_15
            self.__tf_record_val = cfg.TFRECORD_VAL_UPSAMPLE_15
            self.__tf_record_test = cfg.TFRECORD_TEST_UPSAMPLE_15
            self.__count_json = cfg.COUNT_JSON_UPSAMPLE_15
            self.__dictionary = cfg.DICT_CLASSES_15
            self.__reverse_dictionary = cfg.REVERSE_DICT_CLASSES_15
            self.__num_bottom_classes = 11
        elif name == 'downsample_15':
            self.__data_dir = cfg.DATA_DIR_15
            self.__tf_record_train = cfg.TFRECORD_TRAIN_DOWNSAMPLE_15
            self.__tf_record_val = cfg.TFRECORD_VAL_DOWNSAMPLE_15
            self.__tf_record_test = cfg.TFRECORD_TEST_DOWNSAMPLE_15
            self.__count_json = cfg.COUNT_JSON_UPSAMPLE_15
            self.__dictionary = cfg.DICT_CLASSES_15
            self.__reverse_dictionary = cfg.REVERSE_DICT_CLASSES_15
            self.__num_bottom_classes = 11
        elif name == 'original_31':
            self.__data_dir = cfg.DATA_DIR_31
            self.__tf_record_train = cfg.TFRECORD_TRAIN_ORIGINAL_31
            self.__tf_record_val = cfg.TFRECORD_VAL_ORIGINAL_31
            self.__tf_record_test = cfg.TFRECORD_TEST_ORIGINAL_31
            self.__count_json = cfg.COUNT_JSON_ORIGINAL_31
            self.__dictionary = cfg.DICT_CLASSES_31
            self.__reverse_dictionary = cfg.REVERSE_DICT_CLASSES_31
            self.__num_bottom_classes = 27
        elif name == 'upsample_31':
            self.__data_dir = cfg.DATA_DIR_31
            self.__tf_record_train = cfg.TFRECORD_TRAIN_UPSAMPLE_31
            self.__tf_record_val = cfg.TFRECORD_VAL_UPSAMPLE_31
            self.__tf_record_test = cfg.TFRECORD_TEST_UPSAMPLE_31
            self.__count_json = cfg.COUNT_JSON_UPSAMPLE_31
            self.__dictionary = cfg.DICT_CLASSES_31
            self.__reverse_dictionary = cfg.REVERSE_DICT_CLASSES_31
            self.__num_bottom_classes = 27
        elif name == 'downsample_31':
            self.__data_dir = cfg.DATA_DIR_31
            self.__tf_record_train = cfg.TFRECORD_TRAIN_DOWNSAMPLE_31
            self.__tf_record_val = cfg.TFRECORD_VAL_DOWNSAMPLE_31
            self.__tf_record_test = cfg.TFRECORD_TEST_DOWNSAMPLE_31
            self.__count_json = cfg.COUNT_JSON_DOWNSAMPLE_31
            self.__dictionary = cfg.DICT_CLASSES_31
            self.__reverse_dictionary = cfg.REVERSE_DICT_CLASSES_31
            self.__num_bottom_classes = 27
        elif name == 'original_4':
            self.__data_dir = cfg.DATA_DIR_4
            self.__tf_record_train = cfg.TFRECORD_TRAIN_ORIGINAL_4
            self.__tf_record_val = cfg.TFRECORD_VAL_ORIGINAL_4
            self.__tf_record_test = cfg.TFRECORD_TEST_ORIGINAL_4
            self.__count_json = cfg.COUNT_JSON_ORIGINAL_4
            self.__dictionary = cfg.DICT_CLASSES_4
            self.__reverse_dictionary = cfg.REVERSE_DICT_CLASSES_4
            self.__num_bottom_classes = 3
        elif name == 'upsample_4':
            self.__data_dir = cfg.DATA_DIR_4
            self.__tf_record_train = cfg.TFRECORD_TRAIN_DOWNSAMPLE_4
            self.__tf_record_val = cfg.TFRECORD_VAL_DOWNSAMPLE_4
            self.__tf_record_test = cfg.TFRECORD_TEST_DOWNSAMPLE_4
            self.__count_json = cfg.COUNT_JSON_DOWNSAMPLE_4
            self.__dictionary = cfg.DICT_CLASSES_4
            self.__reverse_dictionary = cfg.REVERSE_DICT_CLASSES_4
            self.__num_bottom_classes = 3
        else:
            raise ValueError('No valid dataset')

    """ Only allow read on variables"""
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

    def __get_dictionary(self):
        return self.__dictionary

    def __get_reverse_dictionary(self):
        return self.__reverse_dictionary

    def __get_num_bottom_classes(self):
        return self.__num_bottom_classes

    name = property(__get_name)
    num_classes = property(__get_num_classes)
    data_dir = property(__get_data_dir)
    tf_record_train = property(__get_tf_record_train)
    tf_record_val = property(__get_tf_record_val)
    tf_record_test = property(__get_tf_record_test)
    count_json = property(__get_count_json)
    dictionary = property(__get_dictionary)
    reverse_dictionary = property(__get_reverse_dictionary)
    num_bottom_classes = property(__get_num_bottom_classes)


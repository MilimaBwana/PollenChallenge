import sys
import os
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2 as cv
from collections import Counter, defaultdict
import json
import math
import matplotlib.ticker as mtick

""" Syspath needs to include parent directory "pollen classification" and "Code" to find sibling 
modules and database."""
file_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + "/"
sys.path.append(file_path)

from config import dataset_config
from preprocessing import tf_record_reader, pipeline_ops
from preprocessing.cv2_augmentation import __rotate, __flip_left_right, __flip_up_down, __crop, __noise


def main():
    #plot_sample_images('original_31')
    #plot_one_image_per_class(dataset_name='original_31')
    #plot_img_sizes(dataset_name='original_15')
    #plot_class_distribution(dataset_name='original_4')
    #class_distribution_in_tf_records_to_json(dataset_name='original_31')
    #plot_compared_class_distribution(dataset_name_1='original_31', dataset_name_2='original_15')
    #class_distribution_to_json(dataset_name='original_4')
    #plot_augmentations(dataset_name='original_4')
    read_filenames_from_tf_record('original_4', 'val')


def plot_sample_images(dataset_name='original_15'):
    """
    Plots 12 images out of the given dataset. Tf record of this dataset needs to exists.
    @param dataset_name: name of the dataset
    @return Nothing.
    """
    params = {'dataset': dataset_config.Dataset(dataset_name),
              'input_shape': (224, 224, 3),
              'batch_size': 64,
              'augmentation': True,
              'augmentation_techniques': ['rotate'],
              'colormap': 'none',
              }

    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    tf.config.experimental.set_memory_growth(physical_devices[0], True)  # TF only uses needed GPU RAM, not all

    ds = tf_record_reader.read(tf.estimator.ModeKeys.TRAIN, params)
    samples = ds.take(1)

    for idx, sample in enumerate(samples):
        image, label, _ = sample
        ax = []
        fig = plt.figure(figsize=(15, 15))
        start = random.randint(0, params['batch_size'] - 12)
        for i in range(12):
            ax.append(fig.add_subplot(3, 4, i + 1))
            ax[-1].title.set_text(params['dataset'].reverse_dictionary[label[i + start].numpy()])
            img = np.squeeze(image[i + start].numpy())
            plt.imshow(img, cmap='gray')

        """ Actually displaying the plot if you are not in interactive mode. """
        plt.show()


def plot_one_image_per_class(dataset_name='original_15'):
    """
    Creates a plot with one random image from each class in the given dataset.
    @param dataset_name: Name of the dataset
    @return Nothing
    """
    images_to_plot = []
    dataset = dataset_config.Dataset(dataset_name)
    labeled_subdirectories = pipeline_ops.get_labeled_valid_subdirectories(dataset.name, dataset.data_dir,
                                                                           dictionary=dataset.dictionary,
                                                                           consider_misspelled=False)
    labeled_file_paths = pipeline_ops.get_labeled_png_paths(labeled_subdirectories)
    classes = set([x[1] for x in labeled_file_paths])

    for cl in classes:
        all_files_class = [f[0] for f in labeled_file_paths if f[1] == cl]
        images_to_plot.append([np.random.choice(a=all_files_class, size=1)[0], cl])

    size = len(classes) * 0.5
    fig = plt.figure(figsize=(size, size))
    ax = []
    for idx, imgfile in enumerate(images_to_plot):
        ax.append(fig.add_subplot((len(images_to_plot) // 4) + 1, 4, idx + 1))
        ax[-1].title.set_text(dataset.reverse_dictionary[imgfile[1]])
        img = cv.imread(str(imgfile[0]), 0)
        plt.imshow(img, cmap='gray')

    """ Actually displaying the plot if you are not in interactive mode. """
    plt.show()


def plot_img_sizes(dataset_name='original_15'):
    """
    Plots a 2D-heatmap with the distribution of the image sizes in the dataset.
    @param dataset_name: name of the dataset
    @return: Nothing.
    """
    dataset = dataset_config.Dataset(dataset_name)
    valid_subdirectories = pipeline_ops.get_valid_subdirectories(dataset.name, dataset.data_dir,
                                                                 dictionary=dataset.dictionary,
                                                                 consider_misspelled=True)
    all_files = pipeline_ops.get_png_paths(valid_subdirectories)
    all_images_size = [cv.imread(str(f), 0).shape for f in all_files]
    max_shape = max(all_images_size)
    min_shape = min(all_images_size)
    print('Max img shape', max_shape)
    print('Min img shape', min_shape)

    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    H, xedges, yedges, im = ax.hist2d(x=[c[0] for c in all_images_size], y=[c[1] for c in all_images_size], bins=7,
                                      cmap=plt.cm.Blues)
    """ Use white text if squares are dark; otherwise black text. """
    threshold = H.max() * 3. / 4
    for i in range(H.shape[1]):
        for j in range(H.shape[0]):
            color = 'white' if H[i, j] > threshold else 'black'
            ax.text(xedges[j] + 3, yedges[i] + 15, int(H[j, i]), ha="left", va="top", color=color)
            # Note: Texts are shifted because annotations dont fit into their bins

    plt.colorbar(im, ax=ax)
    ax.set_title('Histogram of image sizes')
    plt.xticks(xedges.astype(int))
    plt.yticks(yedges.astype(int))
    plt.xlabel("Width", axes=ax)
    plt.ylabel("Height", axes=ax)
    fig.tight_layout()
    plt.show(aspect='auto')


def plot_class_distribution(dataset_name='original_15'):
    """
    Plots the class distribution in the dataset as a bar plot. Also prints the number of samples
    of the class with the most samples, the least samples and the imbalance quotient.
    @param dataset_name: Name of the dataset.
    @return Nothing
    """
    dataset = dataset_config.Dataset(dataset_name)
    labeled_subdirectories = pipeline_ops.get_labeled_valid_subdirectories(dataset.name, dataset.data_dir,
                                                                           dictionary=dataset.dictionary,
                                                                           consider_misspelled=True)
    labeled_files = pipeline_ops.get_labeled_png_paths(labeled_subdirectories)
    counter_classes = Counter([c[1] for c in labeled_files])
    print('Max number of samples', max(counter_classes.values()))
    print('Min number of samples', min(counter_classes.values()))
    print('Imbalance quotient', max(counter_classes.values()) / min(counter_classes.values()))

    labels = [dataset.reverse_dictionary[k] for k in counter_classes.keys()]
    values = counter_classes.values()
    num_samples = sum(values)
    """ Sort in lexical order. """
    labels, values = [list(a) for a in zip(*sorted(zip(labels, values), key=lambda x: x[0]))]

    size = len(counter_classes) * 0.5
    fig = plt.figure(figsize=(size, 8))
    ax = [fig.add_subplot(1, 1, 1)]

    axis_font = {'fontname': 'Arial', 'size': '15'}
    plt.rc('font', size=14)
    plt.rc('axes', titlesize=14, labelsize=14)


    for label in (ax[-1].get_xticklabels() + ax[-1].get_yticklabels()):
        label.set_fontsize(15)

    ax[-1].set_title('Class frequency in dataset', fontdict={'fontname': 'Arial', 'size': '16'})
    rects = ax[-1].bar(labels, values, align='center', alpha=0.5)
    plt.xticks(rotation=45)
    vals = ax[-1].get_yticks()
    ax[-1].get_yaxis().set_major_formatter(mtick.PercentFormatter())
    ax[-1].set_yticklabels(['{:,.0%}'.format(np.round(x/num_samples, 3) ) for x in vals])
    #ax[-1].set_yticklabels(['{:,.0%}'.format(x) for x in vals])
    ax[-1].set_xticks(np.arange(len(labels)))
    ax[-1].set_xlabel("Classes", fontdict=axis_font)
    ax[-1].set_ylabel("Frequency [%]", fontdict=axis_font)

    #_autolabel(ax, rects, [np.round(x / num_samples, 3) for x in values])
    _autolabel(ax, rects, [x for x in values])

    plt.show(aspect='auto')


def plot_compared_class_distribution(dataset_name_1='original_4', dataset_name_2='original_15'):
    """
    Compares the class distribution between 'dataset1' and 'dataset2' with a bar plot. Also prints the number of samples
    of the class with the most samples, the least samples and the imbalance quotient for both datasets.
    Currently works for 'upsample' and 'original'.
    @param dataset_name_1: name of the first dataset.
    @param dataset_name_2. name of the second dataset.
    @return Nothing.
    """

    width = 0.3
    dataset_1 = dataset_config.Dataset(dataset_name_1)
    dataset_2 = dataset_config.Dataset(dataset_name_2)
    labeled_subdirectories = pipeline_ops.get_labeled_valid_subdirectories(dataset_1.name, dataset_1.data_dir,
                                                                           dictionary=dataset_1.dictionary,
                                                                           consider_misspelled=True)
    labeled_files = pipeline_ops.get_labeled_png_paths(labeled_subdirectories)
    counter_classes_ds1 = Counter([c[1] for c in labeled_files])
    print(dataset_1.name)
    print('\tMax number of samples', max(counter_classes_ds1.values()))
    print('\tMin number of samples', min(counter_classes_ds1.values()))
    print('\tImbalance quotient ', max(counter_classes_ds1.values()) / min(counter_classes_ds1.values()))

    labeled_subdirectories = pipeline_ops.get_labeled_valid_subdirectories(dataset_2.name, dataset_2.data_dir,
                                                                           dictionary=dataset_2.dictionary,
                                                                           consider_misspelled=True)
    labeled_files = pipeline_ops.get_labeled_png_paths(labeled_subdirectories)
    counter_classes_ds2 = Counter([c[1] for c in labeled_files])

    print(dataset_2.name)
    print('\tMax number of samples', max(counter_classes_ds2.values()))
    print('\tMin number of samples', min(counter_classes_ds2.values()))
    print('\tImbalance quotient ', max(counter_classes_ds2.values()) / min(counter_classes_ds2.values()))

    labels_ds1 = [dataset_1.reverse_dictionary[k] for k in counter_classes_ds1.keys()]
    values_ds1 = list(counter_classes_ds1.values())
    labels_ds2 = [dataset_2.reverse_dictionary[k] for k in counter_classes_ds2.keys()]
    values_ds2 = list(counter_classes_ds2.values())

    labels = set(labels_ds1 + labels_ds2)
    values = []

    """ Merge the values of the datasets. """
    for label in labels:
        tmp = []
        if label in labels_ds1:
            index = labels_ds1.index(label)
            tmp.append(values_ds1[index])
        else:
            tmp.append(0)

        if label in labels_ds2:
            index = labels_ds2.index(label)
            tmp.append(values_ds2[index])
        else:
            tmp.append(0)

        values.append(tmp)

    """ Sort in lexical order. """
    size = len(labels) * 0.5
    labels, values = [list(a) for a in zip(*sorted(zip(labels, values), key=lambda x: x[0]))]
    fig = plt.figure(figsize=(size, 8))

    ax = [fig.add_subplot(1, 1, 1)]
    ax[-1].set_title('Class frequency in dataset')
    rects_ds1 = ax[-1].bar(np.arange(len(labels)) - width / 2, [x[0] for x in values], width, label=dataset_1.name)
    rects_ds2 = ax[-1].bar(np.arange(len(labels)) + width / 2, [x[1] for x in values], width, label=dataset_2.name)
    ax[-1].set_xticks(np.arange(len(labels)))
    ax[-1].set_xlabel("Classes")
    ax[-1].set_ylabel("Frequency")
    plt.xticks(rotation=45)
    ax[-1].set_xticklabels(labels)

    # _autolabel(ax, rects_ds1, rects_ds1)
    # _autolabel(ax[-1], rects_ds2)

    plt.legend()
    plt.show()


def class_distribution_to_json(dataset_name='original_15'):
    """
    Save the class distribution in the 'dataset_name' in a json file.
    @param dataset_name: Name of the dataset.
    @return Nothing
    """
    dataset = dataset_config.Dataset(dataset_name)
    labeled_subdirectories = pipeline_ops.get_labeled_valid_subdirectories(dataset.name, dataset.data_dir,
                                                                           dictionary=dataset.dictionary,
                                                                           consider_misspelled=True)
    labeled_files = pipeline_ops.get_labeled_png_paths(labeled_subdirectories)
    """ ClassIDs as keys."""
    counter_classes = Counter([c[1] for c in labeled_files])
    """ Class names as keys. """
    counter_classes = {dataset.reverse_dictionary[k]: counter_classes[k] for k in counter_classes.keys()}

    with open(os.path.join(dataset.data_dir, dataset.name + "_class_distribution" + ".json"), "w") as file:
        """ Saves occurences per class with class name as key"""
        json.dump(counter_classes, file)


def class_distribution_in_tf_records_to_json(dataset_name='original_15'):
    """
    Save the class distribution in the dataset, split into train, val and test parts, in a json file.
    Collects the class distribution from the respective tf records.
    @param dataset_name: Name of the dataset.
    @return Nothing
    """

    params = {'dataset': dataset_config.Dataset(dataset_name),
              'input_shape': (84, 84, 3),
              'batch_size': 256,
              'augmentation': True,
              'augmentation_techniques': [],
              'colormap': 'none',
              }

    ds_train = tf_record_reader.read(tf.estimator.ModeKeys.TRAIN, params)
    dictionary = defaultdict()

    count_dict = defaultdict()
    for sample_batch in ds_train:
        _, labels, _ = sample_batch
        labels = labels.numpy()
        """ Merge dictionaries"""
        count_dict = Counter(labels) + Counter(count_dict)

    count_dict = {params['dataset'].reverse_dictionary[k]: count_dict[k] for k in count_dict.keys()}
    dictionary['train'] = count_dict

    count_dict = defaultdict()
    ds_val = tf_record_reader.read(tf.estimator.ModeKeys.EVAL, params)
    for sample_batch in ds_val:
        _, labels, _ = sample_batch
        labels = labels.numpy()
        """ Merge dictionaries"""
        count_dict = Counter(labels) + Counter(count_dict)

    count_dict = {params['dataset'].reverse_dictionary[k]: count_dict[k] for k in count_dict.keys()}
    dictionary['val'] = count_dict

    count_dict = defaultdict()
    ds_test = tf_record_reader.read(tf.estimator.ModeKeys.PREDICT, params)
    for sample_batch in ds_test:
        _, labels, _ = sample_batch
        labels = labels.numpy()
        """ Merge dictionaries"""
        count_dict = Counter(labels) + Counter(count_dict)

    count_dict = {params['dataset'].reverse_dictionary[k]: count_dict[k] for k in count_dict.keys()}
    dictionary['test'] = count_dict

    with open(os.path.join(params['dataset'].data_dir,
                           params['dataset'].name + "_class_distribution_in_tfrecords" + ".json"), "w") as file:
        """ Saves occurences per class with class name as key"""
        json.dump(dictionary, file)


def plot_augmentations(dataset_name='original_15'):
    """
    Displays all dataset_manipulator augmentations on a random image from the given dataset.
    @param dataset_name: Name of the dataset.
    @return Nothing
    """
    dataset = dataset_config.Dataset(dataset_name)
    valid_subdirectories = pipeline_ops.get_valid_subdirectories(dataset.name, dataset.data_dir,
                                                                 dictionary=dataset.dictionary,
                                                                 consider_misspelled=True)
    all_files = pipeline_ops.get_png_paths(valid_subdirectories)
    rnd_image_file = np.random.choice(all_files, size=1)
    img = cv.imread(str(np.squeeze(rnd_image_file)), 0)

    augmentation_functions = [__rotate, __flip_left_right, __flip_up_down, __crop, __noise]

    nb_rows = math.ceil((len(augmentation_functions) + 1) / 3)
    fig = plt.figure(figsize=(8, 8))
    ax = [fig.add_subplot(nb_rows, 3, 1)]

    ax[-1].set_title('Original image')
    plt.imshow(img, cmap='gray')

    for idx, augmentation in enumerate(augmentation_functions):
        ax.append(fig.add_subplot(nb_rows, 3, idx + 2))
        ax[-1].set_title(augmentation.__name__[2:].capitalize())
        augmented_img = augmentation(img)
        plt.imshow(augmented_img, cmap='gray')

    plt.show()


def read_filenames_from_tf_record(dataset_name, mode='train'):

    params = {'dataset': dataset_config.Dataset(dataset_name),
              'input_shape': (224, 224, 3),
              'batch_size': 64,
              'augmentation': False,
              'augmentation_techniques': [''],
              'colormap': 'none',
              }

    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    tf.config.experimental.set_memory_growth(physical_devices[0], True)  # TF only uses needed GPU RAM, not all

    if mode == 'train':
        ds = tf_record_reader.read(tf.estimator.ModeKeys.TRAIN, params)
        save_path = params['dataset'].data_dir + '/train_samples_list.txt'
    elif mode == 'val':
        ds = tf_record_reader.read(tf.estimator.ModeKeys.EVAL, params)
        save_path = params['dataset'].data_dir + '/val_samples_list.txt'
    elif mode == 'test':
        ds = tf_record_reader.read(tf.estimator.ModeKeys.PREDICT, params)
        save_path = params['dataset'].data_dir + '/test_samples_list.txt'
    else:
        raise ValueError('No valid mode')

    with open(save_path, 'w') as file:
        for sample in ds:
            _, _, file_paths = sample
            file_paths = file_paths.numpy()

            string_filepaths ="\n".join(path.decode('utf-8') for path in file_paths)
            file.write(string_filepaths)
            file.write('\n')




def _autolabel(ax, rects, values):
    """Attach a text label above each bar in *rects*, displaying its height."
    @param ax: matplotlib axis.
    @param rects: bars to label.
    @param values: int value to be displayed as a label.
    @return Nothing
    """
    for rect, value in zip(rects, values):
        #text_height = rect.get_height() - rect.get_height()*0.2
        text_height = rect.get_height() + 5
        ax[-1].text(rect.get_x() + rect.get_width() / 2, text_height, value,
                    ha='center', va='bottom')


if __name__ == "__main__":
    main()

import sys
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2 as cv
from collections import Counter, defaultdict
import json
import math
import matplotlib.ticker as mtick

""" Syspath needs to include parent directory "pollen_classification" to find sibling modules ."""
file_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + "/"
sys.path.append(file_path)

from config import dataset_config
from preprocessing import tf_record_reader, pipeline_ops
from preprocessing.cv2_augmentation import __rotate, __flip_left_right, __flip_up_down, __crop, __noise


def main():
    # plot_sample_images('original_31')
    plot_one_image_per_class(dataset_name='original_4')
    # plot_img_sizes_heatmap(dataset_name='original_15')
    # plot_img_sizes_hexbin(dataset_name='original_15')
    # plot_class_distribution(dataset_name='original_4')
    # class_distribution_in_tf_records_to_json(dataset_name='upsample_31')
    # plot_compared_class_distribution(dataset_name_1='original_31', dataset_name_2='original_15')
    # class_distribution_to_json(dataset_name='original_4')
    # plot_augmentations(dataset_name='original_4')
    # plot_image_embedding_and_resize_with_pad('original_31')
    # read_filenames_from_tf_record('original_15', 'val')


def plot_sample_images(dataset_name='original_15', number_samples=10):
    """
    Plots number_samples images out of the given dataset. Train tf record of this dataset needs to exists.
    @param dataset_name: name of the dataset
    @param number_samples: number of images to plot
    @return Nothing.
    """
    params = {'dataset': dataset_config.Dataset(dataset_name),
              'input_shape': (224, 224, 3),
              'augmentation': False,
              'augmentation_techniques': ['rotate'],
              'colormap': 'none',
              }

    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    tf.config.experimental.set_memory_growth(physical_devices[0], True)  # TF only uses needed GPU RAM, not all

    ds = tf_record_reader.read(tf.estimator.ModeKeys.PREDICT, params) # Batch size of predict dataset is alqays 1
    samples = ds.take(number_samples)
    rows = (number_samples - 1) // 5 + 1
    width = 10
    height = 2 * rows
    ax = []
    fig = plt.figure(figsize=(width, height))

    for idx, sample in enumerate(samples):
        image, label, _ = sample
        image = tf.squeeze(image)
        label = tf.squeeze(label)
        ax.append(fig.add_subplot(rows, 5, idx + 1))
        ax[-1].title.set_text(params['dataset'].dictionary[label.numpy()])
        img = np.squeeze(image.numpy())
        plt.imshow(img, cmap='gray')

    # Actually displaying the plot if you are not in interactive mode.
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
                                                                           dictionary=dataset.reverse_dictionary,
                                                                           consider_misspelled=False)
    labeled_file_paths = pipeline_ops.get_labeled_png_paths(labeled_subdirectories)
    classes = set([x[1] for x in labeled_file_paths])

    for cl in classes:
        all_files_class = [f[0] for f in labeled_file_paths if f[1] == cl]
        images_to_plot.append([np.random.choice(a=all_files_class, size=1)[0], cl])

    if dataset.num_classes > 4:
        width = 10
        height = 2 * len(classes) // 5
        fig = plt.figure(figsize=(width, height))
        ax = []
        for idx, imgfile in enumerate(images_to_plot):
            ax.append(fig.add_subplot(((len(images_to_plot) - 1) // 5) + 1, 5, idx + 1))
            ax[-1].title.set_text(dataset.dictionary[imgfile[1]])
            img = cv.imread(str(imgfile[0]), 0)
            plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    else:
        width = 8
        height =3
        fig = plt.figure(figsize=(width, height))
        ax = []
        for idx, imgfile in enumerate(images_to_plot):
            ax.append(fig.add_subplot(((len(images_to_plot) - 1) // 4) + 1, 4, idx + 1))
            ax[-1].title.set_text(dataset.dictionary[imgfile[1]])
            img = cv.imread(str(imgfile[0]), 0)
            plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))

    # Actually displaying the plot if you are not in interactive mode.
    plt.show()


def plot_img_sizes_heatmap(dataset_name='original_15'):
    """
    Plots a 2D-heatmap with the distribution of the image sizes in the dataset.
    @param dataset_name: name of the dataset
    @return: Nothing.
    """
    dataset = dataset_config.Dataset(dataset_name)
    valid_subdirectories = pipeline_ops.get_valid_subdirectories(dataset.name, dataset.data_dir,
                                                                 dictionary=dataset.reverse_dictionary,
                                                                 consider_misspelled=True)
    all_files = pipeline_ops.get_png_paths(valid_subdirectories)
    all_images_size = [cv.imread(str(f), 0).shape for f in all_files]
    max_shape = all_images_size[np.argmax(np.asarray([x[0] * x[1] for x in all_images_size]))]
    min_shape = all_images_size[np.argmin(np.asarray([x[0] * x[1] for x in all_images_size]))]
    print('Max img shape', max_shape)
    print('Min img shape', min_shape)

    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    H, xedges, yedges, im = ax.hist2d(x=[c[0] for c in all_images_size], y=[c[1] for c in all_images_size], bins=7,
                                      cmap=plt.cm.Blues)
    # Use white text if squares are dark; otherwise black text.
    threshold = H.max() * 3. / 4
    for i in range(H.shape[1]):
        for j in range(H.shape[0]):
            color = 'white' if H[i, j] > threshold else 'black'
            ax.text(xedges[j] + 3, yedges[i] + 15, int(H[j, i]), ha="left", va="top", color=color)
            # Note: Texts are shifted because annotations dont fit into their bins

    legend = plt.colorbar(im, ax=ax)
    legend.set_label('Frequency')
    ax.set_title('Image size distribution in '+ dataset_name)
    plt.xticks(xedges.astype(int))
    plt.yticks(yedges.astype(int))
    plt.xlabel("Width", axes=ax)
    plt.ylabel("Height", axes=ax)
    fig.tight_layout()
    plt.show(aspect='auto')


def plot_img_sizes_hexbin(dataset_name='original_15'):
    """
    Plots a 2D hexagonal binning plot with the distribution of the image sizes in the dataset.
    @param dataset_name: name of the dataset
    @return: Nothing.
    """
    dataset = dataset_config.Dataset(dataset_name)
    valid_subdirectories = pipeline_ops.get_valid_subdirectories(dataset.name, dataset.data_dir,
                                                                 dictionary=dataset.reverse_dictionary,
                                                                 consider_misspelled=True)
    all_files = pipeline_ops.get_png_paths(valid_subdirectories)
    all_images_size = [cv.imread(str(f), 0).shape for f in all_files]
    max_shape = all_images_size[np.argmax(np.asarray([x[0] * x[1] for x in all_images_size]))]
    min_shape = all_images_size[np.argmin(np.asarray([x[0] * x[1] for x in all_images_size]))]
    print('Max img shape', max_shape)
    print('Min img shape', min_shape)

    fig, ax = plt.subplots()
    # ax.set_aspect("equal")
    hb = ax.hexbin(x=[c[0] for c in all_images_size], y=[c[1] for c in all_images_size], gridsize=30, bins='log',
                   cmap='jet')
    # Use white text if squares are dark; otherwise black text.

    legend = plt.colorbar(hb, ax=ax)
    legend.set_label('Frequency')
    ax.set_title('Image size distribution in ' + dataset_name)
    plt.xlabel("Width", axes=ax)
    plt.ylabel("Height", axes=ax)
    fig.tight_layout()
    plt.show()


def plot_class_distribution(dataset_name='original_15'):
    """
    Plots the class distribution in the dataset as a bar plot. Also prints the number of samples
    of the class with the most samples, the least samples and the imbalance quotient.
    @param dataset_name: Name of the dataset.
    @return Nothing
    """
    dataset = dataset_config.Dataset(dataset_name)
    labeled_subdirectories = pipeline_ops.get_labeled_valid_subdirectories(dataset.name, dataset.data_dir,
                                                                           dictionary=dataset.reverse_dictionary,
                                                                           consider_misspelled=True)
    labeled_files = pipeline_ops.get_labeled_png_paths(labeled_subdirectories)
    counter_classes = Counter([c[1] for c in labeled_files])
    print('Max number of samples', max(counter_classes.values()))
    print('Min number of samples', min(counter_classes.values()))
    print('Imbalance quotient', max(counter_classes.values()) / min(counter_classes.values()))

    labels = [dataset.dictionary[k] for k in counter_classes.keys()]
    values = counter_classes.values()
    num_samples = sum(values)
    # Sort in lexical order.
    labels, values = [list(a) for a in zip(*sorted(zip(labels, values), key=lambda x: x[0]))]

    # Labels on top of bars
    plt.rc('font', size=15)
    # plt.rc('axes', titlesize=12, labelsize=11)

    if dataset.num_classes < 20:

        size = len(counter_classes) * 2
        fig = plt.figure(figsize=(size, 6))
        ax = [fig.add_subplot(1, 1, 1)]

        ax[-1].set_title('Class frequency in ' + dataset.name, fontdict={'fontname': 'DejaVu Sans', 'size': '16'})
        rects = ax[-1].bar(labels, values, align='center', alpha=0.5)
        plt.xticks(rotation=45)
        vals = ax[-1].get_yticks()
        ax[-1].get_yaxis().set_major_formatter(mtick.PercentFormatter())
        ax[-1].set_yticklabels(['{:,.0%}'.format(np.round(x / num_samples, 3)) for x in vals])
        # ax[-1].set_yticklabels(['{:,.0%}'.format(x) for x in vals])
        ax[-1].set_xticks(np.arange(len(labels)))

        axis_font = {'fontname': 'DejaVu Sans', 'size': '18'}
        ax[-1].set_xlabel("Classes", fontdict=axis_font)
        ax[-1].set_ylabel("Frequency [%]", fontdict=axis_font)

        # _autolabel(ax, rects, [np.round(x / num_samples, 3) for x in values])
        _autolabel_vertical_bars(ax, rects, [x for x in values])
        # Remove padding at left and right
        plt.margins(x=0.01)

    else:

        size = len(counter_classes) * 0.5
        fig = plt.figure(figsize=(10, size))
        ax = [fig.add_subplot(1, 1, 1)]

        ax[-1].set_title('Class frequency in' + dataset_name, fontdict={'fontname': 'DejaVu Sans', 'size': '16'})
        y_pos = np.arange(len(labels))
        rects = ax[-1].barh(y_pos, values, align='center', alpha=0.5)
        ax[-1].invert_yaxis()
        ax[-1].set_yticks(y_pos)
        ax[-1].set_yticklabels(labels)

        #Set labels in original_31, that are not in original_15, to red.
        # dataset_small_dict = dataset_config.Dataset('original_15').reverse_dictionary
        # idx_labels_colored = [True if label in dataset_small_dict else False for label in labels]
        # idx_labels_colored= [i for i, x in enumerate(idx_labels_colored) if x]
        # [x.set_color('red') for idx, x in enumerate(ax[-1].get_yticklabels()) if idx not in idx_labels_colored]

        vals = ax[-1].get_xticks()
        ax[-1].set_xticklabels(['{:,.0%}'.format(np.round(x / num_samples, 3)) for x in vals])

        axis_font = {'fontname': 'DejaVu Sans', 'size': '15'}
        ax[-1].set_xlabel("Frequency [%]", fontdict=axis_font)
        ax[-1].set_ylabel("Classes", fontdict=axis_font)

        _autolabel_horizontal_bars(ax, rects, [x for x in values])

        # Remove padding at top and bottom
        plt.margins(y=0.01)

    for label in (ax[-1].get_xticklabels() + ax[-1].get_yticklabels()):
        # tick labels on x and y axis
        label.set_fontsize(16)

    plt.show()


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
                                                                           dictionary=dataset_1.reverse_dictionary,
                                                                           consider_misspelled=True)
    labeled_files = pipeline_ops.get_labeled_png_paths(labeled_subdirectories)
    counter_classes_ds1 = Counter([c[1] for c in labeled_files])
    print(dataset_1.name)
    print('\tMax number of samples', max(counter_classes_ds1.values()))
    print('\tMin number of samples', min(counter_classes_ds1.values()))
    print('\tImbalance quotient ', max(counter_classes_ds1.values()) / min(counter_classes_ds1.values()))

    labeled_subdirectories = pipeline_ops.get_labeled_valid_subdirectories(dataset_2.name, dataset_2.data_dir,
                                                                           dictionary=dataset_2.reverse_dictionary,
                                                                           consider_misspelled=True)
    labeled_files = pipeline_ops.get_labeled_png_paths(labeled_subdirectories)
    counter_classes_ds2 = Counter([c[1] for c in labeled_files])

    print(dataset_2.name)
    print('\tMax number of samples', max(counter_classes_ds2.values()))
    print('\tMin number of samples', min(counter_classes_ds2.values()))
    print('\tImbalance quotient ', max(counter_classes_ds2.values()) / min(counter_classes_ds2.values()))

    labels_ds1 = [dataset_1.dictionary[k] for k in counter_classes_ds1.keys()]
    values_ds1 = list(counter_classes_ds1.values())
    labels_ds2 = [dataset_2.dictionary[k] for k in counter_classes_ds2.keys()]
    values_ds2 = list(counter_classes_ds2.values())

    labels = set(labels_ds1 + labels_ds2)
    values = []

    # Merge the values of the datasets.
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

    # Sort in lexical order.
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
                                                                           dictionary=dataset.reverse_dictionary,
                                                                           consider_misspelled=True)
    labeled_files = pipeline_ops.get_labeled_png_paths(labeled_subdirectories)
    # ClassIDs as keys.
    counter_classes = Counter([c[1] for c in labeled_files])
    # Class names as keys.
    counter_classes = {dataset.dictionary[k]: counter_classes[k] for k in counter_classes.keys()}

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
        # Merge dictionaries
        count_dict = Counter(labels) + Counter(count_dict)

    count_dict = {params['dataset'].dictionary[k]: count_dict[k] for k in count_dict.keys()}
    dictionary['train'] = count_dict

    count_dict = defaultdict()
    ds_val = tf_record_reader.read(tf.estimator.ModeKeys.EVAL, params)
    for sample_batch in ds_val:
        _, labels, _ = sample_batch
        labels = labels.numpy()
        # Merge dictionaries
        count_dict = Counter(labels) + Counter(count_dict)

    count_dict = {params['dataset'].dictionary_[k]: count_dict[k] for k in count_dict.keys()}
    dictionary['val'] = count_dict

    count_dict = defaultdict()
    ds_test = tf_record_reader.read(tf.estimator.ModeKeys.PREDICT, params)
    for sample_batch in ds_test:
        _, labels, _ = sample_batch
        labels = labels.numpy()
        # Merge dictionaries
        count_dict = Counter(labels) + Counter(count_dict)

    count_dict = {params['dataset'].dictionary[k]: count_dict[k] for k in count_dict.keys()}
    dictionary['test'] = count_dict

    with open(os.path.join(params['dataset'].data_dir,
                           params['dataset'].name + "_class_distribution_in_tfrecords" + ".json"), "w") as file:
        # Saves occurences per class with class name as key
        json.dump(dictionary, file)


def plot_augmentations(dataset_name='original_15'):
    """
    Displays all dataset_manipulator augmentations on a random image from the given dataset.
    @param dataset_name: Name of the dataset.
    @return Nothing
    """
    dataset = dataset_config.Dataset(dataset_name)
    valid_subdirectories = pipeline_ops.get_valid_subdirectories(dataset.name, dataset.data_dir,
                                                                 dictionary=dataset.reverse_dictionary,
                                                                 consider_misspelled=True)
    all_files = pipeline_ops.get_png_paths(valid_subdirectories)
    rnd_image_file = np.random.choice(all_files, size=1)
    img = cv.imread(str(np.squeeze(rnd_image_file)), 0)

    augmentation_functions = [__rotate, __flip_left_right, __flip_up_down, __crop, __noise]

    nb_rows = math.ceil((len(augmentation_functions) + 1) / 3)
    fig = plt.figure(figsize=(12, 8))
    ax = [fig.add_subplot(nb_rows, 3, 1)]

    axis_font = {'fontname': 'DejaVu Sans', 'size': '18'}
    ax[-1].set_title('Original image', fontdict=axis_font)
    plt.imshow(img, cmap='gray')

    for idx, augmentation in enumerate(augmentation_functions):
        ax.append(fig.add_subplot(nb_rows, 3, idx + 2))
        ax[-1].set_title(augmentation.__name__[2:].capitalize(), fontdict=axis_font)
        augmented_img = augmentation(img)
        plt.imshow(augmented_img, cmap='gray')

    plt.show()


def plot_image_embedding_and_resize_with_pad(dataset_name='original_31'):
    """
    Shows the preprocessing of images in terms of size using two sample images. The smaller image is
    embedded with black pixels, the larger image is resized with padding.
    @param dataset_name: Name of the dataset. Should be 'original_31', because the range of image sizes
        is wider in this dataset.
    @return: Nothing
    """
    dataset = dataset_config.Dataset(dataset_name)
    valid_subdirectories = pipeline_ops.get_valid_subdirectories(dataset.name, dataset.data_dir,
                                                                 dictionary=dataset.reverse_dictionary,
                                                                 consider_misspelled=True)

    input_shape = (224, 224, 1)

    all_files = pipeline_ops.get_png_paths(valid_subdirectories)
    # Exclude no pollen class, because it has some strange images.
    all_files = [x for x in all_files if 'NoPollen' not in str(x)]
    all_images = [cv.imread(str(f), 0) for f in all_files]

    img_max_shape = all_images[np.argmax(np.asarray([x.shape[0] * x.shape[1] for x in all_images]))]
    img_min_shape = all_images[np.argmin(np.asarray([x.shape[0] * x.shape[1] for x in all_images]))]

    tf_img_max_shape = tf.expand_dims(tf.convert_to_tensor(img_max_shape), axis=-1)
    tf_img_min_shape = tf.expand_dims(tf.convert_to_tensor(img_min_shape), axis=-1)

    # Embed smaller image into a black background.
    offset_height = (input_shape[0] - tf.shape(tf_img_min_shape)[0]) // 2
    offset_width = (input_shape[1] - tf.shape(tf_img_min_shape)[1]) // 2
    img_embedded = tf.image.pad_to_bounding_box(tf_img_min_shape, offset_height=offset_height,
                                                offset_width=offset_width,
                                                target_height=input_shape[0],
                                                target_width=input_shape[1]).numpy()

    # Resize while keeping the aspect ratio the same without distortion.
    img_resized = tf.image.resize_with_pad(tf_img_max_shape, target_height=input_shape[0],
                                           target_width=input_shape[1]).numpy()

    fig = plt.figure(figsize=(6, 6))
    axis_font = {'fontname': 'DejaVu Sans', 'size': '16'}

    # Add original small image
    ax = [fig.add_subplot(2, 2, 1)]
    ax[-1].set_title('Original image', fontdict=axis_font)
    plt.imshow(img_min_shape, cmap='gray')

    # Add embedded small image
    ax.append(fig.add_subplot(2, 2, 2))
    ax[-1].set_title('Embedded image', fontdict=axis_font)
    plt.imshow(img_embedded, cmap='gray')

    # Add original small image
    ax.append(fig.add_subplot(2, 2, 3))
    ax[-1].set_title('Original image', fontdict=axis_font)
    plt.imshow(img_max_shape, cmap='gray')

    ax.append(fig.add_subplot(2, 2, 4))
    ax[-1].set_title('Resized image', fontdict=axis_font)
    plt.imshow(img_resized, cmap='gray')

    plt.show()


def read_filenames_from_tf_record(dataset_name, mode='train'):
    """
    Saves all filenames from a tf record to a txt-file.
    Txt-file is saved at data_dir of the dataset.
    @param dataset_name: Name of the dataset.
    @param mode: whether to save filenames from train, val or test tf record.
    @return: Nothing.
    """

    params = {'dataset': dataset_config.Dataset(dataset_name),
              'input_shape': (224, 224, 3),
              'batch_size': 64,
              'augmentation': False,
              'augmentation_techniques': [''],
              'colormap': 'none',
              }

    # physical_devices = tf.config.experimental.list_physical_devices('GPU')
    # assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    # tf.config.experimental.set_memory_growth(physical_devices[0], True)  # TF only uses needed GPU RAM, not all

    if mode == 'train':
        ds = tf_record_reader.read(tf.estimator.ModeKeys.TRAIN, params)
        save_path = params['dataset'].data_dir + '/' + dataset_name + '_train_samples_list.txt'
    elif mode == 'val':
        ds = tf_record_reader.read(tf.estimator.ModeKeys.EVAL, params)
        save_path = params['dataset'].data_dir + '/' + dataset_name + '_val_samples_list.txt'
    elif mode == 'test':
        ds = tf_record_reader.read(tf.estimator.ModeKeys.PREDICT, params)
        save_path = params['dataset'].data_dir + '/' + dataset_name + '_test_samples_list.txt'
    else:
        raise ValueError('No valid mode')

    with open(save_path, 'w') as file:
        for sample in ds:
            _, _, file_paths = sample
            file_paths = file_paths.numpy()

            string_filepaths = "\n".join(path.decode('utf-8') for path in file_paths)
            file.write(string_filepaths)
            file.write('\n')


def _autolabel_vertical_bars(ax, rects, values):
    """Attach a text label above each vertical bar in *rects*, displaying its height."
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


def _autolabel_horizontal_bars(ax, rects, values):
    """Attach a text label above each horizontal bar in *rects*, displaying its height."
    @param ax: matplotlib axis.
    @param rects: bars to label.
    @param values: int value to be displayed as a label.
    @return Nothing
    """
    for rect, value in zip(rects, values):
        text_width = rect.get_width() + 200
        ax[-1].text(text_width, rect.get_y() + rect.get_height() * 2 / 3, value,
                    ha='center', va='bottom')


if __name__ == "__main__":
    main()

import os
import platform
import re

DICT_CLASSES_31 = {
    'Alnus': 0,  # Erle
    'Apiaceae': 1,  # Doldenblütler
    'Artemisia': 2,  # Beifuss
    'Betula': 3,  # Birke
    'Cannabaceae': 4,  # Hanf
    'Carpinus': 5,  # Hainbuche
    'Castanea': 6,  # Esskastanie
    'Chenopodiaceae': 7,  # Gänsefuss
    'Corylus': 8,  # Hasel
    'Cupressaceae': 9,  # Zypressen
    'Cyperaceae': 10,  # Sauergras
    'Fagus': 11,  # Buche
    'Fraxinus': 12,  # Esche
    'Juglans': 13,  # Walnuss
    'Larix': 14,  # # Lärche
    'Papaveraceae': 15,  # Mohngewächse
    'Picea': 16,  # Fichte
    'Pinaceae': 17,  # Kiefer
    'Plantago': 18,  # Wegerich
    'Platanus': 19,  # Platane
    'Poaceae': 20,  # Gras
    'Populus': 21,  # Papel
    'Quercus': 22,  # Eiche
    'Rumex': 23,  # Ampfer
    'Salix': 24,  # Weide
    'Taxus': 25,  # Eibe
    'Tilia': 26,  # Linde
    'Ulmus': 27,  # Ulme
    'Urticaceae': 28,  # Brennnessel
    'Spores': 29,  # Sporen
    'NoPollen': 30  # keine Pollen
}

REVERSE_DICT_CLASSES_31 = {
    0: 'Alnus',  # Erle
    1: 'Apiaceae',  # Doldenblütler
    2: 'Artemisia',  # Beifuss
    3: 'Betula',  # Birke
    4: 'Cannabaceae',  # Hanf
    5: 'Carpinus',  # Hainbuche
    6: 'Castanea',  # Esskastanie
    7: 'Chenopodiaceae',  # Gänsefuss
    8: 'Corylus',  # Hasel
    9: 'Cupressaceae',  # Zypressen
    10: 'Cyperaceae',  # Sauergras
    11: 'Fagus',  # Buche
    12: 'Fraxinus',  # Esche
    13: 'Juglans',  # Walnuss
    14: 'Larix',  # # Lärche
    15: 'Papaveraceae',  # Mohngewächse
    16: 'Picea',  # Fichte
    17: 'Pinaceae',  # Kiefer
    18: 'Plantago',  # Wegerich
    19: 'Platanus',  # Platane
    20: 'Poaceae',  # Gras
    21: 'Populus',  # Papel
    22: 'Quercus',  # Eiche
    23: 'Rumex',  # Ampfer
    24: 'Salix',  # Weide
    25: 'Taxus',  # Eibe
    26: 'Tilia',  # Linde
    27: 'Ulmus',  # Ulme
    28: 'Urticaceae',  # Brennnessel
    29: 'Spores',  # Sporen
    30: 'NoPollen'  # keine Pollen
}

DICT_CLASSES_15 = {'Alnus': 0,  # Erle
                   'Betula': 1,  # Birke
                   'Carpinus': 2,  # Hainbuche
                   'Corylus': 3,  # Hasel
                   'Fagus': 4,  # Buche
                   'Fraxinus': 5,  # Esche
                   'Plantago': 6,  # Wegerich
                   'Poaceae': 7,  # Gras
                   'Populus': 8,  # Papel
                   'Quercus': 9,  # Eiche
                   'Salix': 10,  # Weide
                   'Taxus': 11,  # Eibe
                   'Tilia': 12,  # Linde
                   'Ulmus': 13,  # Ulme
                   'Urticaceae': 14,  # Brennnessel
                   }

REVERSE_DICT_CLASSES_15 = {0: 'Alnus',  # Erle
                           1: 'Betula',  # Birke
                           2: 'Carpinus',  # Hainbuche
                           3: 'Corylus',  # Hasel
                           4: 'Fagus',  # Buche
                           5: 'Fraxinus',  # Esche
                           6: 'Plantago',  # Wegerich
                           7: 'Poaceae',  # Gras
                           8: 'Populus',  # Papel
                           9: 'Quercus',  # Eiche
                           10: 'Salix',  # Weide
                           11: 'Taxus',  # Eibe
                           12: 'Tilia',  # Linde
                           13: 'Ulmus',  # Ulme
                           14: 'Urticaceae',  # Brennnessel
                           }
0
DICT_CLASSES_4 = {'Corylus': 0,  # Corylus avellana, well-developed pollen grains
                  'AnomalousCorylus': 1,  # Corylus avellana, anomalous pollen grains
                  'Alnus': 2,  # Erle
                  'Debris': 3,  # NoPollen
                  }

REVERSE_DICT_CLASSES_4 = {0: 'Corylus',  # Corylus avellana, well-developed pollen grains
                          1: 'AnomalousCorylus',  # Corylus avellana, anomalous pollen grains
                          2: 'Alnus',  # Erle
                          3: 'Debris',  # NoPollen
                          }

"""
# Pollen_dir_matcher without renaming
POLLEN_DIR_MATCHER = {
        #Wrong word : Correct Word 
        "Betula": "Betula",
        "Betula1": "Betula",
        "Populus": "Populus",
        "Coyrlus": "Corylus",
        "Tilia": "Tilia",
        "Urticaceae": "Urticaceae"
    }
"""
# Pollen_dir_matcher with renamingw
POLLEN_DIR_MATCHER = {
    # Wrong word : Correct Word
    "Betula2": "Betula",
    "Betula1": "Betula",
    "Cupressaceae1": "Cupressaceae",
    "CupressaceaeCupressaceae": "Cupressaceae",
    "Chenopodium": "Chenopodiaceae",
    "Coyrlus": "Corylus",
    "Pinus": "Pinaceae",
    "Populus1": "Populus",
    "Sauergras": "Cyperaceae",
    "Tilia1": "Tilia",
    "Urticaceae1": "Urticaceae"
}

""" Data directory contains the folders with different classes. """
# TODO: to be edited, if new dataset
if re.match('eihw-gpu[0-4]', platform.node()):
    """ Eihw cluster """
    DATA_DIR = "/nas/student/JakobSchaefer/data"
    DATA_DIR_15 = DATA_DIR + '/batch_from_20190128'
    DATA_DIR_31 = DATA_DIR + '/batch_from_20190128'
    DATA_DIR_4 = DATA_DIR + '/italian_data'
    LOG_DIR = '/nas/student/JakobSchaefer/pollenanalysis2.0/src/pollen_classification/nn_models/logs'
    CHKPT_DIR = '/nas/student/JakobSchaefer/pollenanalysis2.0/src/pollen_classification/nn_models/tf_ckpt'
else:
    """ local computer (Windows or Ubuntu) """
    DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "data")
    DATA_DIR_15 = DATA_DIR + '/augsburg_data'
    DATA_DIR_31 = DATA_DIR + '/augsburg_data'
    DATA_DIR_4 = DATA_DIR + '/italian_data'
    LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "nn_models/logs")
    CHKPT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "nn_models/tf_ckpt")

UPSAMPLE_DIR_EXTENSION = '_upsample'
UPSAMPLE_NAME = 'upsample'
DOWNSAMPLE_NAME = 'downsample'
ORIGINAL_NAME = 'original'

CSV_TRAIN_LABELS_15 = DATA_DIR_15 + '/labels_train.csv'
CSV_VAL_LABELS_15 = DATA_DIR_15 + '/labels_devel.csv'
CSV_TEST_LABELS_15 = DATA_DIR_15 + '/labels_test.csv'


TRAIN_SPLIT = 0.6
VAL_SPLIT = 0.2

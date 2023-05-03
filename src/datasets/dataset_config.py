from os.path import join

_BASE_DATA_PATH = "../../data"

dataset_config = {
    'mnist': {
        'path': join(_BASE_DATA_PATH, 'mnist'),
        'normalize': ((0.1307,), (0.3081,)),
        # Use the next 3 lines to use MNIST with a 3x32x32 input
        # 'extend_channel': 3,
        # 'pad': 2,
        # 'normalize': ((0.1,), (0.2752,))    # values including padding
    },
    'svhn': {
        'path': join(_BASE_DATA_PATH, 'svhn'),
        'resize': (224, 224),
        'crop': None,
        'flip': False,
        'normalize': ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    },
    'cifar100': {
        'path': join(_BASE_DATA_PATH, 'cifar100'),
        'resize': None,
        'pad': 4,
        'crop': 32,
        'flip': True,
        'normalize': ((0.5071, 0.4866, 0.4409), (0.2009, 0.1984, 0.2023))
    },
    'cifar100_icarl': {
        'path': join(_BASE_DATA_PATH, 'cifar100'),
        'resize': None,
        'pad': 4,
        'crop': 32,
        'flip': True,
        'normalize': ((0.5071, 0.4866, 0.4409), (0.2009, 0.1984, 0.2023)),
        'class_order': [
            68, 56, 78, 8, 23, 84, 90, 65, 74, 76, 40, 89, 3, 92, 55, 9, 26, 80, 43, 38, 58, 70, 77, 1, 85, 19, 17, 50,
            28, 53, 13, 81, 45, 82, 6, 59, 83, 16, 15, 44, 91, 41, 72, 60, 79, 52, 20, 10, 31, 54, 37, 95, 14, 71, 96,
            98, 97, 2, 64, 66, 42, 22, 35, 86, 24, 34, 87, 21, 99, 0, 88, 27, 18, 94, 11, 12, 47, 25, 30, 46, 62, 69,
            36, 61, 7, 63, 75, 5, 32, 4, 51, 48, 73, 93, 39, 67, 29, 49, 57, 33
        ]
    },
    'vggface2': {
        'path': join(_BASE_DATA_PATH, 'VGGFace2'),
        'resize': 256,
        'crop': 224,
        'flip': True,
        'normalize': ((0.5199, 0.4116, 0.3610), (0.2604, 0.2297, 0.2169))
    },
    'imagenet_256': {
        'path': join(_BASE_DATA_PATH, 'ILSVRC12_256'),
        'resize': None,
        'crop': 224,
        'flip': True,
        'normalize': ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    },
    'imagenet_subset': {
        'path': join(_BASE_DATA_PATH, 'ILSVRC12_256'),
        'resize': None,
        'crop': 224,
        'flip': True,
        'normalize': ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        'class_order': [
            68, 56, 78, 8, 23, 84, 90, 65, 74, 76, 40, 89, 3, 92, 55, 9, 26, 80, 43, 38, 58, 70, 77, 1, 85, 19, 17, 50,
            28, 53, 13, 81, 45, 82, 6, 59, 83, 16, 15, 44, 91, 41, 72, 60, 79, 52, 20, 10, 31, 54, 37, 95, 14, 71, 96,
            98, 97, 2, 64, 66, 42, 22, 35, 86, 24, 34, 87, 21, 99, 0, 88, 27, 18, 94, 11, 12, 47, 25, 30, 46, 62, 69,
            36, 61, 7, 63, 75, 5, 32, 4, 51, 48, 73, 93, 39, 67, 29, 49, 57, 33
        ]
    },
    'imagenet_32_reduced': {
        'path': join(_BASE_DATA_PATH, 'ILSVRC12_32'),
        'resize': None,
        'pad': 4,
        'crop': 32,
        'flip': True,
        'normalize': ((0.481, 0.457, 0.408), (0.260, 0.253, 0.268)),
        'class_order': [
            472, 46, 536, 806, 547, 976, 662, 12, 955, 651, 492, 80, 999, 996, 788, 471, 911, 907, 680, 126, 42, 882,
            327, 719, 716, 224, 918, 647, 808, 261, 140, 908, 833, 925, 57, 388, 407, 215, 45, 479, 525, 641, 915, 923,
            108, 461, 186, 843, 115, 250, 829, 625, 769, 323, 974, 291, 438, 50, 825, 441, 446, 200, 162, 373, 872, 112,
            212, 501, 91, 672, 791, 370, 942, 172, 315, 959, 636, 635, 66, 86, 197, 182, 59, 736, 175, 445, 947, 268,
            238, 298, 926, 851, 494, 760, 61, 293, 696, 659, 69, 819, 912, 486, 706, 343, 390, 484, 282, 729, 575, 731,
            530, 32, 534, 838, 466, 734, 425, 400, 290, 660, 254, 266, 551, 775, 721, 134, 886, 338, 465, 236, 522, 655,
            209, 861, 88, 491, 985, 304, 981, 560, 405, 902, 521, 909, 763, 455, 341, 905, 280, 776, 113, 434, 274, 581,
            158, 738, 671, 702, 147, 718, 148, 35, 13, 585, 591, 371, 745, 281, 956, 935, 346, 352, 284, 604, 447, 415,
            98, 921, 118, 978, 880, 509, 381, 71, 552, 169, 600, 334, 171, 835, 798, 77, 249, 318, 419, 990, 335, 374,
            949, 316, 755, 878, 946, 142, 299, 863, 558, 306, 183, 417, 64, 765, 565, 432, 440, 939, 297, 805, 364, 735,
            251, 270, 493, 94, 773, 610, 278, 16, 363, 92, 15, 593, 96, 468, 252, 699, 377, 95, 799, 868, 820, 328, 756,
            81, 991, 464, 774, 584, 809, 844, 940, 720, 498, 310, 384, 619, 56, 406, 639, 285, 67, 634, 792, 232, 54,
            664, 818, 513, 349, 330, 207, 361, 345, 279, 549, 944, 817, 353, 228, 312, 796, 193, 179, 520, 451, 871,
            692, 60, 481, 480, 929, 499, 673, 331, 506, 70, 645, 759, 744, 459]
    },
    'mirage_generic': {
        'path': join(_BASE_DATA_PATH, 'mirage_generic',
                     'dataset_100pkt_5f_payload_df_exact_noNullver_no0load_noMaps_b8829cf2.parquet'),
        # 'class_order': [12, 11, 1, 9, 4, 2, 0, 7, 5, 3, 8, 10, 6], # Used for Algorithm type FSL (fine-tuning etc.)
        'fs_split': {  # Used for Model type FSL (meta-learning etc.).
            'train_classes': [8, 16, 1, 9, 21, 19, 35, 27, 5, 17, 3, 38, 7, 4, 36, 10, 39, 2, 26, 23, 30, 22, 20, 37],
            # (>1000, top-24)
            'val_classes': [0, 32, 6, 15, 18, 25, 31, 34],  # (>1000, bottom-8)
            'test_classes': [14, 24, 13, 12, 28, 29, 33, 11]  # (<1000, worst-8)
        },
    },
    'mirage_generic_nosni_obf': {
        'path': join(_BASE_DATA_PATH, 'mirage_generic',
                     'dataset_100pkt_5f_payload_df_exact_noNullver_no0load_noMaps_noSniObfRand_b8829cf2.parquet'),
        # 'class_order': [12, 11, 1, 9, 4, 2, 0, 7, 5, 3, 8, 10, 6], # Used for Algorithm type FSL (fine-tuning etc.)
        'fs_split': {  # Used for Model type FSL (meta-learning etc.).
            'train_classes': [8, 16, 1, 9, 21, 19, 35, 27, 5, 17, 3, 38, 7, 4, 36, 10, 39, 2, 26, 23, 30, 22, 20, 37],
            # (>1000, top-24)
            'val_classes': [0, 32, 6, 15, 18, 25, 31, 34],  # (>1000, bottom-8)
            'test_classes': [14, 24, 13, 12, 28, 29, 33, 11]  # (<1000, worst-8)
        },
    },
    'mirage_generic_nosni_drp': {
        'path': join(_BASE_DATA_PATH, 'mirage_generic',
                     'dataset_100pkt_5f_payload_df_exact_noNullver_no0load_noMaps_noSniRem_b8829cf2.parquet'),
        # 'class_order': [12, 11, 1, 9, 4, 2, 0, 7, 5, 3, 8, 10, 6], # Used for Algorithm type FSL (fine-tuning etc.)
        'fs_split': {  # Used for Model type FSL (meta-learning etc.).
            'train_classes': [8, 16, 1, 9, 21, 19, 35, 27, 5, 17, 3, 38, 7, 4, 36, 10, 39, 2, 26, 23, 30, 22, 20, 37],
            # (>1000, top-24)
            'val_classes': [0, 32, 6, 15, 18, 25, 31, 34],  # (>1000, bottom-8)
            'test_classes': [14, 24, 13, 12, 28, 29, 33, 11]  # (<1000, worst-8)
        },
    },
    'mirage_generic_nosni_pad': {
        'path': join(_BASE_DATA_PATH, 'mirage_generic',
                     'dataset_100pkt_5f_payload_df_exact_noNullver_no0load_noMaps_noSniPadRand_b8829cf2.parquet'),
        # 'class_order': [12, 11, 1, 9, 4, 2, 0, 7, 5, 3, 8, 10, 6], # Used for Algorithm type FSL (fine-tuning etc.)
        'fs_split': {  # Used for Model type FSL (meta-learning etc.).
            'train_classes': [8, 16, 1, 9, 21, 19, 35, 27, 5, 17, 3, 38, 7, 4, 36, 10, 39, 2, 26, 23, 30, 22, 20, 37],
            # (>1000, top-24)
            'val_classes': [0, 32, 6, 15, 18, 25, 31, 34],  # (>1000, bottom-8)
            'test_classes': [14, 24, 13, 12, 28, 29, 33, 11]  # (<1000, worst-8)
        },
    },
    'mirage_generic_cl': {
        'path': join(_BASE_DATA_PATH, 'mirage_generic_cl',
                     'dataset_100pkt_5f_payload_df_exact_noNullver_no0load_noMaps_pseudolabels_nodegclust_merged_b8829cf2_fsl_24nf.parquet'),
        'fs_split': {  
            'train_classes': [
                39, 22, 1, 87, 23, 14, 57, 49, 8, 29, 76, 27, 88, 73, 44,
                6, 4, 59, 91, 56, 96, 65, 21, 11, 19, 48, 46, 2, 26, 40,
                103, 13, 30, 99, 51, 25, 12, 94, 74, 10, 43, 71, 69, 31,
                92, 100, 28, 61, 80, 33, 24, 41, 20, 97, 50, 98, 66, 5,
                79, 55, 101, 70, 62, 53, 42, 89, 3, 17, 93, 82, 52, 32,45,
                95, 9, 15, 64, 7, 60, 90, 63, 54, 58, 16, 75, 102, 72, 81
            ], # top-24 mirage_generic classes clustered to 89 classes
            'val_classes': [0, 84, 18, 38, 47, 68, 83, 86],  
            'test_classes': [37, 67, 36, 35, 77, 78, 85, 34] 
        }
    },
    'iot23': {  # class order
        'path': join(_BASE_DATA_PATH, 'iot23',
                     'dataset_20p_6f_576b_obf.parquet'),
        'class_order': [12, 11, 1, 9, 4, 2, 0, 7, 5, 3, 8, 10, 6],  # Used for Algorithm type FSL (fine-tuning etc.)
        'fs_split': {  # Used for Model type FSL (meta-learning etc.)
            'train_classes': [12, 11, 1, 9, 4],  # PartOfAHorizontalPortScan, Okiru, Benign, DDoS, C&C-HeartBeat
            'val_classes': [2, 0, 7, 5],  # C&C, Attack, C&C-PartOfAHorizontalPortScan, C&C-HeartBeat-Attack
            'test_classes': [3, 8, 10, 6]
            # (<100) C&C-FileDownload, C&C-Torii, FileDownload, C&C-HeartBeat-FileDownload
        }
    },
    'iot23b': {  # class order
        'path': join(_BASE_DATA_PATH, 'iot23b',
                     'dataset_20p_6f_576b_obf_bin.parquet'),
        'class_order': [0, 1],  # Used for Algorithm type FSL (fine-tuning etc.)
        'fs_split': {  # Used for Model type FSL (meta-learning etc.)
            'train_classes': [0],  # Benign
            'val_classes': [],
            'test_classes': [1]  # Malicious
        }
    },
    'appclassnet': {  # class order
        'path': join(_BASE_DATA_PATH, 'appclassnet',
                     'dataset_20p_2f.parquet'),
        'class_order': list(range(200)),  # Used for Algorithm type FSL (fine-tuning etc.)
        'fs_split': {  # Used for Model type FSL (meta-learning etc.)
            'train_classes': list(range(128)),
            'val_classes': list(range(128, 160)),
            'test_classes': list(range(160, 200))
        }
    }
}

# Add missing keys:
for dset in dataset_config.keys():
    for k in ['resize', 'pad', 'crop', 'normalize', 'class_order', 'extend_channel']:
        if k not in dataset_config[dset].keys():
            dataset_config[dset][k] = None
    if 'flip' not in dataset_config[dset].keys():
        dataset_config[dset]['flip'] = False

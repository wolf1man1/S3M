print("DEBUG: Script Launching... (If you see this, python is working)", flush=True)
import os
import numpy as np
import scipy.io
import h5py
import torch
import math
from tqdm import tqdm
import sys
import subprocess
import glob
import tifffile
import shutil
import gc
import re

# ================= Configuration =================
# Input Paths (Kaggle Environment)
# Blacklist (Copied from consts.py for standalone execution)
BLACKLIST_FILES = [
    "Health_hyper_167.tif", "Health_hyper_26.tif", "Other_hyper_22.tif", "Health_hyper_12.tif",
    "Health_hyper_23.tif", "Other_hyper_149.tif", "Health_hyper_76.tif", "Health_hyper_38.tif",
    "Other_hyper_174.tif", "Health_hyper_34.tif", "Other_hyper_122.tif", "Health_hyper_153.tif",
    "Other_hyper_113.tif", "Other_hyper_121.tif", "Other_hyper_64.tif", "Other_hyper_163.tif",
    "Health_hyper_67.tif", "Other_hyper_102.tif", "Other_hyper_31.tif", "Other_hyper_155.tif",
    "Other_hyper_50.tif", "Other_hyper_26.tif", "Other_hyper_160.tif"
]

DATASET_CONFIG = {
    'HySpecNet224Ext': {
        'path': [
            '/kaggle/input/notebooks/xishengfeng/hyspecnet11k-03',
            '/kaggle/input/notebooks/xishengfeng/hyspecnet-11k-04',
            '/kaggle/input/notebooks/xishengfeng/hyspecnet11k-02',
            '/kaggle/input/notebooks/xishengfeng/hyspecnet11k-02.',
            '/kaggle/input/notebooks/xishengfeng/hyspecnet11k-05',
            '/kaggle/input/notebooks/xishengfeng/hyspecnet11k-06',
            '/kaggle/input/notebooks/xishengfeng/hyspecnet-11k-07',
            '/kaggle/input/notebooks/xishengfeng/hyspecnet11k-08',
            '/kaggle/input/notebooks/xishengfeng/hyspecnet11k-09'
        ],
        'is_directory': True,
        'key': 'none',
        'scale_factor': 10000.0,
        'normalize': False,
        'enabled': False
    },
    'HanChuan': {
        'path': '/kaggle/input/whu-hyperspectral-dataset/WHU-Hi-HanChuan/WHU_Hi_HanChuan.mat',
        'key': 'WHU_Hi_HanChuan',
        'scale_factor': 1.0,
        'normalize': True  # WHU-Hi is Radiance, need 0-1
    },
    'HongHu': {
        'path': '/kaggle/input/whu-hyperspectral-dataset/WHU-Hi-HongHu/WHU_Hi_HongHu.mat',
        'key': 'WHU_Hi_HongHu',
        'scale_factor': 1.0,
        'normalize': True
    },
    'LongKou': {
        'path': '/kaggle/input/whu-hyperspectral-dataset/WHU-Hi-LongKou/WHU_Hi_LongKou.mat',
        'key': 'WHU_Hi_LongKou',
        'scale_factor': 1.0,
        'normalize': True
    },
    'Salinas': {
        'path': [
            '/kaggle/input/hyperspectral-image-classification-data-collection/datasets/sa/Salinas.mat',
            '/kaggle/input/datasets/kinggleliu/hyperspectral-image-classification-data-collection/datasets/sa/Salinas.mat'
        ],
        'key': 'salinas',
        'scale_factor': 1.0,  # Raw DN can be very large (e.g. >9000)
        'normalize': True
    },
    'Indian_pines': {
        'path': [
            '/kaggle/input/hyperspectral-image-classification-data-collection/datasets/ip/Indian_pines.mat',
            '/kaggle/input/datasets/kinggleliu/hyperspectral-image-classification-data-collection/datasets/ip/Indian_pines.mat'
        ],
        'key': 'indian_pines',
        'scale_factor': 1.0,  # Raw DN can be very large (e.g. >9000)
        'normalize': True
    },
    'Dioni': {
        'path': [
            '/kaggle/input/hyperspectral-image-classification-data-collection/datasets/HyRANK/Dioni.mat',
            '/kaggle/input/datasets/kinggleliu/hyperspectral-image-classification-data-collection/datasets/HyRANK/Dioni.mat'
        ],
        'key': 'ori_data',
        'scale_factor': 1.0,
        'normalize': True
    },
    'Loukia': {
        'path': [
            '/kaggle/input/hyperspectral-image-classification-data-collection/datasets/HyRANK/Loukia.mat',
            '/kaggle/input/datasets/kinggleliu/hyperspectral-image-classification-data-collection/datasets/HyRANK/Loukia.mat'
        ],
        'key': 'ori_data',
        'scale_factor': 1.0,
        'normalize': True
    },
    'HyperLeaf2024': {
        'path': [
            '/kaggle/input/HyperLeaf2024/images',
            '/kaggle/input/competitions/HyperLeaf2024/images'
        ],
        'key': 'none',
        'scale_factor': 65535.0  # uint16 -> 0-1
    },
    'Kochia': {
        'path': [
            '/kaggle/input/notebooks/zetaoxia/notebook4c8082a2b8',
            '/kaggle/input/weedcube-1',
            '/kaggle/input/ragweed',
            '/kaggle/input/canola-weedcube',
            '/kaggle/input/waterhemp',
            '/kaggle/input/notebooka03803a143',
            '/kaggle/input/sugar',
            '/kaggle/input/notebooks/zetaoxia/weedcube-1',
            '/kaggle/input/notebooks/zetaoxia/ragweed',
            '/kaggle/input/notebooks/zetaoxia/canola-weedcube',
            '/kaggle/input/notebooks/zetaoxia/waterhemp',
            '/kaggle/input/notebooks/zetaoxia/notebooka03803a143',
            '/kaggle/input/notebooks/zetaoxia/sugar'
        ],
        'key': 'none',
        'scale_factor': 1.0 # Already 0-1
    },
    'Chikusei': {
        'path': [
            '/kaggle/input/chikusei/HyperspecVNIR_Chikusei_20140729.mat',
            '/kaggle/input/datasets/mingliu123/chikusei/HyperspecVNIR_Chikusei_20140729.mat'
        ],
        'key': 'chikusei',
        'scale_factor': 1.0,
        'normalize': True
    },
    'KSC': {
        'path': [
            '/kaggle/input/kennedy-space-center/KSC.mat',
            '/kaggle/input/datasets/samyabose/kennedy-space-center/KSC.mat'
        ],
        'key': 'KSC',
        'scale_factor': 1.0,
        'normalize': True # Fixes low mean issue
    },
    'HySpecNet': {
        'path': [
            '/kaggle/input/hyspecnet11k-patch1-10',
            '/kaggle/input/datasets/alexanderpjohn/hyspecnet11k-patch1-10'
        ], # Directory
        'key': 'none',
        'scale_factor': 1.0,
        'normalize': False
    },
    'Botswana': {
        'path': [
            '/kaggle/input/botswana/Botswana.mat',
            '/kaggle/input/datasets/dileepangara/botswana/Botswana.mat'
        ],
        'key': 'Botswana',
        'scale_factor': 1.0,
        'normalize': True
    },
    'EarthView-Neon': {
        'path': '/kaggle/input/notebooks/xishengfeng/earthview-neon',
        'key': 'none',
        'scale_factor': 1.0, 
        'normalize': True
    },
    'Moffett': {
        'path': '/kaggle/input/notebooks/xishengfeng/moffett-field/moffett_field/Moffett_HS_RR.mat',
        'key': 'Moffett_HS_RR', 
        'scale_factor': 1.0,
        'normalize': True
    },
    'Cuprite': {
        'path': '/kaggle/input/notebooks/xishengfeng/other/hyperspectral_data/cuprite/Cuprite_S1_F224.img',
        'key': 'none', # File based
        'scale_factor': 1.0, 
        'normalize': True
    },
    'Samson': {
        'path': '/kaggle/input/notebooks/xishengfeng/other/hyperspectral_data/samson/samson_1.img',
        'key': 'none',
        'scale_factor': 1.0,
        'normalize': True,
        # Source often misses .hdr next to .img; generate one in OUTPUT_DIR/temp_headers.
        'needs_header_fix': True
    },
    'Urban': {
        'path': '/kaggle/input/notebooks/xishengfeng/other/hyperspectral_data/urban/Urban_F210.img',
        'key': 'none',
        'scale_factor': 1.0,
        'normalize': True
    },
    'WashingtonDC': {
        'path': '/kaggle/input/notebooks/xishengfeng/other/hyperspectral_data/washington_dc_mall/dc.tif',
        'key': 'none',
        'scale_factor': 1.0,
        'normalize': True,
        'is_directory': False,
        'transpose': (1,2,0)
    },
    'Karlsruhe': {
        'path': '/kaggle/input/notebooks/xishengfeng/hymap-karlsruhe/HyMap_125bands_Karlsruhe2.tif',
        'key': 'none',
        'is_directory': False,
        'scale_factor': 1.0,
        'normalize': True,
        'transpose': (1,2,0)
    },
    'CoCoaSpec': {
        'path': '/kaggle/input/notebooks/xishengfeng/ecos-nord-ginp-uis-cocoaspec',
        'is_directory': True,
        'filter_kw': '.dat', # Scan all .dat files (including hsi_open.dat and hsi_closed.dat)
        'scale_factor': 1.0, 
        'normalize': True,
        'key': 'none',
        'ignore_kws': ['.cache', 'hsi_white', 'hsi_dark'] # Typically white/dark references are ignored if present
    },
    'AgForest': {
        'path': '/kaggle/input/notebooks/xishengfeng/agriculture-forest-hyperspectral-data',
        'is_directory': True,
        # Only process Reflectance Data, specific file suffix
        'filter_kw': '_Hyperspectral_Data.hdr', 
        'scale_factor': 1.0,  # Assuming reflectance is 0-1, or check metadata to scale if 0-10000. 
        'normalize': True,
        # Metadata says "data ignore value = -9999", "Reflectance" usually 0-1 or 0-10000. 
        # If unknown, Robust Norm will handle scaling (0-1).
        'key': 'none',
        'ignore_kws': ['Radiance_Data', 'Subset_1'] # Ignore Radiance, and potentially small subsets if duplicates
    },
    'SpectroFood_Apple': {
        'path': '/kaggle/input/notebooks/xishengfeng/spectrofood-apple',
        'is_directory': True,
        # Apple folder accidentally contains broccoli mats; keep only Apple cube.
        'filter_kw': 'Apple.mat',
        # Apple appears to be high-range DN (global max ~9526), use fixed scaling to [0, ~1].
        'scale_factor': 9526.0,
        'normalize': False,
        'key': 'none'
    },
    'SpectroFood_Broccoli': {
        'path': [
            '/kaggle/input/notebooks/xishengfeng/spectrofood-apple',
            '/kaggle/input/notebooks/xishengfeng/spectrofood-broccoli1',
            '/kaggle/input/notebooks/xishengfeng/spectrofood-broccoli2',
            '/kaggle/input/notebooks/xishengfeng/spectrofood-broccoli3'
        ],
        'is_directory': True,
        'filter_kw': 'Broccoli_',
        'ignore_kws': ['.hdr'],
        'scale_factor': 1.0,
        'normalize': False,
        # Remove rare sensor hot/dead pixels before any stats/resampling.
        'pre_clip_range': (-0.1, 1.5),
        'key': 'none'
    },
    'SpectroFood_Mushroom': {
        'path': '/kaggle/input/notebooks/xishengfeng/spectrofood-mushroom',
        'is_directory': True,
        'filter_kw': '.mat',
        'ignore_kws': ['.hdr'],
        'scale_factor': 1.0,
        # Reflectance-like values are already near [0, 1]; avoid per-image stretch distortion.
        'normalize': False,
        'key': 'none'
    },
    'SpectroFood_FX10': {
        'path': [
            '/kaggle/input/notebooks/xishengfeng/spectrofood-leek1-fx10',
            '/kaggle/input/notebooks/xishengfeng/spectrofood-leek2-fx10'
        ],
        'is_directory': True,
        'filter_kw': '.mat',
        'scale_factor': 1.0,
        # Reflectance-like values are already near [0, 1]; avoid per-image stretch distortion.
        'normalize': False,
        # FX10 mat files are very large; process one file at a time to avoid RAM spikes.
        'num_workers': 1,
        # Only FX10 uses selective MAT variable loading to avoid OOM on huge files.
        'mat_selective_load': True,
        'key': 'none'
    },
    'WeedHSI_Reflectance': {
        'path': [
            '/kaggle/input/notebooks/xishengfeng/mendeley-data/weed_hyperspectral/Weed Species Identification-A Hyperspectral and RGB Dataset with Labeled Data/Day1/Day1',
            '/kaggle/input/notebooks/xishengfeng/mendeley-data/weed_hyperspectral/Weed Species Identification-A Hyperspectral and RGB Dataset with Labeled Data/Day2/Day2',
            '/kaggle/input/notebooks/xishengfeng/mendeley-data/weed_hyperspectral/Weed Species Identification-A Hyperspectral and RGB Dataset with Labeled Data/Day3/Day3',
            '/kaggle/input/notebooks/xishengfeng/mendeley-data/weed_hyperspectral/Weed Species Identification-A Hyperspectral and RGB Dataset with Labeled Data/Day4/Day4',
            '/kaggle/input/notebooks/xishengfeng/mendeley-data/weed_hyperspectral/Weed Species Identification-A Hyperspectral and RGB Dataset with Labeled Data/Day5/Day5',
            '/kaggle/input/notebooks/xishengfeng/mendeley-data/weed_hyperspectral/Weed Species Identification-A Hyperspectral and RGB Dataset with Labeled Data/Day6/Day6'
        ],
        'is_directory': True,
        # Use only plate reflectance cubes under each day folder.
        'filter_kw': 'REFLECTANCE_Plate',
        'ignore_kws': ['.hdr'],
        'scale_factor': 1.0,
        # Provided stats are already reflectance-like; do not apply robust normalization.
        'normalize': False,
        'key': 'none'
    },
    'Cabbage_Eggplant': {
        'path': [
            '/kaggle/input/notebooks/xishengfeng/cabbage-eggplan/cabbage_eggplant/Cabbage_Eggplant_Crops/CE_Reflectance_Data.hdr',
            '/kaggle/input/notebooks/xishengfeng/cabbage-eggplan/cabbage_eggplant/Cabbage_Eggplant_Crops/CE_Reflectance_Data'
        ],
        'is_directory': True,
        # ENVI reflectance cube under this folder (paired with .hdr).
        'filter_kw': '.dat',
        'ignore_kws': ['.hdr'],
        'scale_factor': 1.0,
        # Value range is already reflectance-like (min~0, max~1.87).
        'normalize': False,
        'key': 'none'
    },
    'HyperspectralBlueberries': {
        'path': [
            '/kaggle/input/notebooks/xishengfeng/hyperspectralblueberries-01',
            '/kaggle/input/notebooks/xishengfeng/hyperspectralblueberries-02'
        ],
        'is_directory': True,
        # ENVI BIL cubes with paired .hdr files.
        'filter_kw': '.bil',
        'ignore_kws': ['.hdr', 'WhiteReference', 'DarkReference'],
        # Use per-file HDR metadata (gain/ceiling) for customized calibration.
        'scale_factor': 1.0,
        'normalize': False,
        # Gain differs across batches (5.0 vs 6.0). Map all to reference gain=5.0.
        'apply_gain_correction': True,
        'gain_reference': 5.0,
        # Convert DN to normalized range using each file's HDR ceiling.
        'use_hdr_ceiling': True,
        # Each cube is very large (1600-1800 x 1600 x 462); keep single-thread to avoid RAM spikes.
        'num_workers': 1,
        'key': 'none'
    },
    'PotatoWaterStress': {
        'path': '/kaggle/input/notebooks/xishengfeng/potatowaterstress-zenodo-05',
        'is_directory': True,
        # ENVI dataset: process via .hdr to robustly resolve paired binary.
        'filter_kw': '.hdr',
        'scale_factor': 1.0,
        'normalize': False,
        # Temporarily disable this dataset from processing.
        'enabled': False,
        # Header says: data ignore value = 2
        'nodata_value': 2.0,
        'nodata_tol': 1e-6,
        'num_workers': 1,
        'key': 'none'
    }
}


# Output Path (write patches directly to Kaggle working root for easier download)
OUTPUT_DIR = '/kaggle/working'
HYSPECNET224EXT_DATASET = 'HySpecNet224Ext'
HYSPECNET224EXT_WORKING_CACHE_DIR = OUTPUT_DIR
HYSPECNET224EXT_CACHE_PATTERN = f'{HYSPECNET224EXT_DATASET}_*.pt'

from threading import Lock
PRINT_LOCK = Lock()
PRINTED_DATASETS = set()
PRINTED_WAVELENGTH_MISMATCHES = set()

# Patch Settings
PATCH_SIZE = 32
STRIDE = 32  # Non-overlapping (except edges)

# Nodata handling (EnMAP L2A commonly uses -9999 fill values)
NODATA_VALUE = -9999.0
NODATA_TOL = 1.0
MIN_PATCH_VALID_RATIO = 0.2
HYSPECNET224_BAND_COUNT = 224
# 0-based indices from HySpecNet notes: remove [127-141] and [161-167].
HYSPECNET_WATER_VAPOR_REMOVE_INDICES = list(range(127, 142)) + list(range(161, 168))

# Datasets that should keep their original spectral band count (skip mismatched cubes).
STRICT_BAND_DATASETS = {
    'SpectroFood_Apple',
    'SpectroFood_Broccoli',
    'SpectroFood_Mushroom',
    'SpectroFood_FX10',
    'Cabbage_Eggplant',
    'HyperspectralBlueberries',
    'PotatoWaterStress'
}

# Datasets with reflectance-like values where extreme numeric ranges likely indicate bad reads.
REFLECTANCE_SANITY_DATASETS = {
    'SpectroFood_Broccoli',
    'SpectroFood_Mushroom',
    'SpectroFood_FX10',
    'WeedHSI_Reflectance',
    'Cabbage_Eggplant',
    'PotatoWaterStress'
}

# Worker settings for directory datasets.
# Default is intentionally higher than CPU cores to better hide I/O wait.
NUM_WORKERS = 4
_num_workers_env = os.environ.get("PREPARE_NUM_WORKERS")
if _num_workers_env:
    try:
        NUM_WORKERS = max(1, int(_num_workers_env))
    except ValueError:
        print(f"[WARNING] Invalid PREPARE_NUM_WORKERS='{_num_workers_env}', fallback to {NUM_WORKERS}", flush=True)

# Target S185 Wavelengths (extracted `s185_band_lookup (1).csv`)
TARGET_WAVELENGTHS = np.array([
    450.0, 454.03225806451616, 458.06451612903226, 462.0967741935484, 466.1290322580645, 
    470.16129032258067, 474.19354838709677, 478.2258064516129, 482.258064516129, 486.2903225806452, 
    490.3225806451613, 494.35483870967744, 498.38709677419354, 502.4193548387097, 506.4516129032258, 
    510.48387096774195, 514.516129032258, 518.5483870967741, 522.5806451612904, 526.6129032258065, 
    530.6451612903226, 534.6774193548387, 538.7096774193549, 542.741935483871, 546.7741935483871, 
    550.8064516129032, 554.8387096774194, 558.8709677419355, 562.9032258064516, 566.9354838709678, 
    570.9677419354839, 575.0, 579.0322580645161, 583.0645161290322, 587.0967741935484, 
    591.1290322580645, 595.1612903225806, 599.1935483870968, 603.2258064516129, 607.258064516129, 
    611.2903225806451, 615.3225806451612, 619.3548387096774, 623.3870967741935, 627.4193548387096, 
    631.4516129032259, 635.483870967742, 639.516129032258, 643.5483870967741, 647.5806451612904, 
    651.6129032258065, 655.6451612903226, 659.6774193548388, 663.7096774193549, 667.741935483871, 
    671.7741935483871, 675.8064516129032, 679.8387096774194, 683.8709677419355, 687.9032258064516, 
    691.9354838709678, 695.9677419354839, 700.0, 704.0322580645161, 708.0645161290322, 
    712.0967741935484, 716.1290322580645, 720.1612903225807, 724.1935483870968, 728.2258064516129, 
    732.258064516129, 736.2903225806451, 740.3225806451612, 744.3548387096774, 748.3870967741935, 
    752.4193548387098, 756.4516129032259, 760.483870967742, 764.516129032258, 768.5483870967741, 
    772.5806451612902, 776.6129032258065, 780.6451612903226, 784.6774193548388, 788.7096774193549, 
    792.741935483871, 796.7741935483871, 800.8064516129032, 804.8387096774193, 808.8709677419355, 
    812.9032258064516, 816.9354838709678, 820.9677419354839, 825.0, 829.0322580645161, 
    833.0645161290322, 837.0967741935484, 841.1290322580645, 845.1612903225807, 849.1935483870968, 
    853.2258064516129, 857.258064516129, 861.2903225806451, 865.3225806451612, 869.3548387096774, 
    873.3870967741935, 877.4193548387098, 881.4516129032259, 885.483870967742, 889.516129032258, 
    893.5483870967741, 897.5806451612902, 901.6129032258065, 905.6451612903226, 909.6774193548388, 
    913.7096774193549, 917.741935483871, 921.7741935483871, 925.8064516129032, 929.8387096774193, 
    933.8709677419355, 937.9032258064516, 941.9354838709678, 945.9677419354839, 950.0
])

CHIKUSEI_WAVELENGTHS = [
    363, 368, 373, 378, 383, 388, 394, 399, 404, 409, 414, 419, 425, 430, 435, 440, 445, 450, 455, 
    461, 466, 471, 476, 481, 486, 492, 497, 502, 507, 512, 517, 523, 528, 533, 538, 543, 548, 554, 
    559, 564, 569, 574, 579, 585, 590, 595, 600, 605, 610, 615, 621, 626, 631, 636, 641, 646, 652, 
    657, 662, 667, 672, 677, 683, 688, 693, 698, 703, 708, 714, 719, 724, 729, 734, 739, 744, 750, 
    755, 760, 765, 770, 775, 781, 786, 791, 796, 801, 806, 812, 817, 822, 827, 832, 837, 843, 848, 
    853, 858, 863, 868, 874, 879, 884, 889, 894, 899, 904, 910, 915, 920, 925, 930, 935, 941, 946, 
    951, 956, 961, 966, 972, 977, 982, 987, 992, 997, 1003, 1008, 1013, 1018
] # Replaced below

# Precision HyperLeaf 2024 (206 Bands? No, file has 17 lines of ~12 items = 200+).
# From hyperleaf2024鏁寸悊娉㈡.txt
HYPERLEAF_WAVELENGTHS = [
    397.32, 400.20, 403.09, 405.97, 408.85, 411.74, 414.63, 417.52, 420.40, 423.29, 426.19, 429.08,
    431.97, 434.87, 437.76, 440.66, 443.56, 446.45, 449.35, 452.25, 455.16, 458.06, 460.96, 463.87,
    466.77, 469.68, 472.59, 475.50, 478.41, 481.32, 484.23, 487.14, 490.06, 492.97, 495.89, 498.80,
    501.72, 504.64, 507.56, 510.48, 513.40, 516.33, 519.25, 522.18, 525.10, 528.03, 530.96, 533.89,
    536.82, 539.75, 542.68, 545.62, 548.55, 551.49, 554.43, 557.36, 560.30, 563.24, 566.18, 569.12,
    572.07, 575.01, 577.96, 580.90, 583.85, 586.80, 589.75, 592.70, 595.65, 598.60, 601.55, 604.51,
    607.46, 610.42, 613.38, 616.34, 619.30, 622.26, 625.22, 628.18, 631.15, 634.11, 637.08, 640.04,
    643.01, 645.98, 648.95, 651.92, 654.89, 657.87, 660.84, 663.81, 666.79, 669.77, 672.75, 675.73,
    678.71, 681.69, 684.67, 687.65, 690.64, 693.62, 696.61, 699.60, 702.58, 705.57, 708.57, 711.56,
    714.55, 717.54, 720.54, 723.53, 726.53, 729.53, 732.53, 735.53, 738.53, 741.53, 744.53, 747.54,
    750.54, 753.55, 756.56, 759.56, 762.57, 765.58, 768.60, 771.61, 774.62, 777.64, 780.65, 783.67,
    786.68, 789.70, 792.72, 795.74, 798.77, 801.79, 804.81, 807.84, 810.86, 813.89, 816.92, 819.95,
    822.98, 826.01, 829.04, 832.07, 835.11, 838.14, 841.18, 844.22, 847.25, 850.29, 853.33, 856.37,
    859.42, 862.46, 865.50, 868.55, 871.60, 874.64, 877.69, 880.74, 883.79, 886.84, 889.90, 892.95,
    896.01, 899.06, 902.12, 905.18, 908.24, 911.30, 914.36, 917.42, 920.48, 923.55, 926.61, 929.68,
    932.74, 935.81, 938.88, 941.95, 945.02, 948.10, 951.17, 954.24, 957.32, 960.40, 963.47, 966.55,
    969.63, 972.71, 975.79, 978.88, 981.96, 985.05, 988.13, 991.22, 994.31, 997.40, 1000.49, 1003.58
]

# Precision Chikusei (128 Bands) from HDR
CHIKUSEI_WAVELENGTHS = [
    362.59, 367.75, 372.90, 378.07, 383.23, 388.39, 393.55, 398.71, 403.87, 409.03, 414.19,
    419.36, 424.52, 429.68, 434.84, 440.00, 445.16, 450.32, 455.48, 460.64, 465.80, 470.96, 476.12,
    481.29, 486.45, 491.61, 496.77, 501.93, 507.09, 512.25, 517.41, 522.57, 527.73, 532.89, 538.06,
    543.21, 548.38, 553.54, 558.70, 563.86, 569.02, 574.18, 579.34, 584.50, 589.66, 594.83, 599.99,
    605.14, 610.31, 615.47, 620.63, 625.79, 630.95, 636.11, 641.27, 646.43, 651.59, 656.75, 661.92,
    667.07, 672.24, 677.40, 682.56, 687.72, 692.88, 698.04, 703.20, 708.36, 713.52, 718.68, 723.85,
    729.01, 734.17, 739.33, 744.49, 749.65, 754.81, 759.97, 765.13, 770.29, 775.45, 780.61, 785.78,
    790.94, 796.10, 801.26, 806.42, 811.58, 816.74, 821.90, 827.06, 832.23, 837.38, 842.54, 847.71,
    852.87, 858.03, 863.19, 868.35, 873.51, 878.67, 883.83, 888.99, 894.16, 899.31, 904.48, 909.64,
    914.80, 919.96, 925.12, 930.28, 935.44, 940.60, 945.76, 950.92, 956.09, 961.25, 966.41, 971.57,
    976.73, 981.89, 987.05, 992.21, 997.37, 1002.53, 1007.69, 1012.86, 1018.02
]

# Precision Salinas (224 Bands) - AVIRIS Full
SALINAS_FULL_WAVELENGTHS = [
    360, 369.7, 379.4, 389.1, 398.8, 408.5, 418.2, 427.9, 437.6, 447.3,
    457, 466.7, 476.4, 486.1, 495.8, 505.5, 515.2, 524.9, 534.6, 544.3,
    554, 563.7, 573.4, 583.1, 592.8, 602.5, 612.2, 621.9, 631.6, 641.3,
    651, 660.7, 660, 669.5, 679, 688.5, 698, 707.5, 717, 726.5,
    736, 745.5, 755, 764.5, 774, 783.5, 793, 802.5, 812, 821.5,
    831, 840.5, 850, 859.5, 869, 878.5, 888, 897.5, 907, 916.5,
    926, 935.5, 945, 954.5, 964, 973.5, 983, 992.5, 1002, 1011.5,
    1021, 1030.5, 1040, 1049.5, 1059, 1068.5, 1078, 1087.5, 1097, 1106.5,
    1116, 1125.5, 1135, 1144.5, 1154, 1163.5, 1173, 1182.5, 1192, 1201.5,
    1211, 1220.5, 1230, 1239.5, 1249, 1258.5, 1260, 1270, 1280, 1290,
    1300, 1310, 1320, 1330, 1340, 1350, 1360, 1370, 1380, 1390,
    1400, 1410, 1420, 1430, 1440, 1450, 1460, 1470, 1480, 1490,
    1500, 1510, 1520, 1530, 1540, 1550, 1560, 1570, 1580, 1590,
    1600, 1610, 1620, 1630, 1640, 1650, 1660, 1670, 1680, 1690,
    1700, 1710, 1720, 1730, 1740, 1750, 1760, 1770, 1780, 1790,
    1800, 1810, 1820, 1830, 1840, 1850, 1860, 1870, 1880, 1890,
    1880, 1890, 1900, 1910, 1920, 1930, 1940, 1950, 1960, 1970,
    1980, 1990, 2000, 2010, 2020, 2030, 2040, 2050, 2060, 2070,
    2080, 2090, 2100, 2110, 2120, 2130, 2140, 2150, 2160, 2170,
    2180, 2190, 2200, 2210, 2220, 2230, 2240, 2250, 2260, 2270,
    2280, 2290, 2300, 2310, 2320, 2330, 2340, 2350, 2360, 2370,
    2380, 2390, 2400, 2410, 2420, 2430, 2440, 2450, 2460, 2470,
    2480, 2490, 2500, 2510
]

# Precision HyRANK (176 Bands) - Hyperion
HYRANK_WAVELENGTHS = [
    426.82, 436.99, 447.17, 457.34, 467.52, 477.69, 487.87, 498.04, 508.22, 518.39,
    528.57, 538.74, 548.92, 559.09, 569.27, 579.45, 589.62, 599.80, 609.97, 620.15,
    630.32, 640.50, 650.67, 660.85, 671.02, 681.20, 691.37, 701.55, 711.72, 721.90,
    732.07, 742.25, 752.43, 762.60, 772.78, 782.95, 793.13, 803.30, 813.48, 823.65,
    833.83, 844.00, 854.18, 864.35, 874.53, 884.70, 894.88, 905.05, 915.23, 925.41,
    912.45, 922.54, 932.64, 942.73, 952.82, 962.91, 972.99, 983.08, 993.17, 1003.30,
    1013.30, 1023.40, 1033.49, 1043.59, 1053.69, 1063.79, 1073.89, 1083.99, 1094.09, 1104.19,
    1114.19, 1124.28, 1134.38, 1144.48, 1154.58, 1164.68, 1174.77, 1184.87, 1194.97, 1205.07,
    1215.17, 1225.17, 1235.27, 1245.36, 1255.46, 1265.56, 1275.66, 1285.76, 1295.86, 1305.96,
    1316.05, 1326.05, 1336.15, 1346.25, 1356.35, 1366.45, 1376.55, 1386.65, 1396.74, 1406.84,
    1416.94, 1426.94, 1437.04, 1447.14, 1457.23, 1467.33, 1477.43, 1487.53, 1497.63, 1507.73,
    1517.83, 1527.92, 1537.92, 1548.02, 1558.12, 1568.22, 1578.32, 1588.42, 1598.51, 1608.61,
    1618.71, 1628.81, 1638.81, 1648.90, 1659.00, 1669.10, 1679.20, 1689.30, 1699.40, 1709.50,
    1719.60, 1729.70, 1739.70, 1749.79, 1759.89, 1769.99, 1780.09, 1790.19, 1800.29, 1810.38,
    1820.48, 1830.58, 1840.58, 1850.68, 1860.78, 1870.87, 1880.98, 1891.07, 1901.17, 1911.27,
    1921.37, 1931.47, 1941.57, 1951.57, 1961.66, 1971.76, 1981.86, 1991.96, 2002.06, 2012.15,
    2022.25, 2032.35, 2042.45, 2052.45, 2062.55, 2072.65, 2082.75, 2092.84, 2102.94, 2113.04,
    2123.14, 2133.24, 2143.34, 2153.34, 2163.43, 2173.53, 2183.63, 2193.73, 2203.83, 2213.93,
    2224.03, 2234.12, 2244.22, 2254.22, 2264.32, 2274.42, 2284.52, 2294.61, 2304.71, 2314.81,
    2324.91, 2335.01, 2345.11, 2355.21, 2365.20, 2375.30, 2385.40, 2395.50
]

# Precision Botswana (145 Bands) - Hyperion EO-1
# Band selection: (10-55, 82-97, 102-119, 134-164, 187-220) from 242 original bands
# Exact wavelengths extracted from botswana.txt - using Hyperion sensor documentation
BOTSWANA_WAVELENGTHS = [
    # VNIR B10-B55 (46 bands): 447.17nm - 905.05nm
    447.17, 457.34, 467.52, 477.69, 487.87, 498.04, 508.22, 518.39, 528.57, 538.74,
    548.92, 559.09, 569.27, 579.45, 589.62, 599.80, 609.97, 620.15, 630.32, 640.50,
    650.67, 660.85, 671.02, 681.20, 691.37, 701.55, 711.72, 721.90, 732.07, 742.25,
    752.43, 762.60, 772.78, 782.95, 793.13, 803.30, 813.48, 823.65, 833.83, 844.00,
    854.18, 864.35, 874.53, 884.70, 894.88, 905.05,
    # SWIR B82-B97 (16 bands): 962.91nm - 1114.19nm
    962.91, 972.99, 983.08, 993.17, 1003.30, 1013.30, 1023.40, 1033.49,
    1043.59, 1053.69, 1063.79, 1073.89, 1083.99, 1094.09, 1104.19, 1114.19,
    # SWIR B102-B119 (18 bands): 1164.68nm - 1336.15nm
    1164.68, 1174.77, 1184.87, 1194.97, 1205.07, 1215.17, 1225.17, 1235.27,
    1245.36, 1255.46, 1265.56, 1275.66, 1285.76, 1295.86, 1305.96, 1316.05,
    1326.05, 1336.15,
    # SWIR B134-B164 (31 bands): 1477.43nm - 1780.09nm
    1477.43, 1487.53, 1497.63, 1507.73, 1517.83, 1527.92, 1537.92, 1548.02,
    1558.12, 1568.22, 1578.32, 1588.42, 1598.51, 1608.61, 1618.71, 1628.81,
    1638.81, 1648.90, 1659.00, 1669.10, 1679.20, 1689.30, 1699.40, 1709.50,
    1719.60, 1729.70, 1739.70, 1749.79, 1759.89, 1769.99, 1780.09,
    # SWIR B187-B220 (34 bands): 2022.25nm - 2355.21nm
    2022.25, 2032.35, 2042.45, 2052.45, 2062.55, 2072.65, 2082.75, 2092.84,
    2102.94, 2113.04, 2123.14, 2133.24, 2143.34, 2153.34, 2163.43, 2173.53,
    2183.63, 2193.73, 2203.83, 2213.93, 2224.03, 2234.12, 2244.22, 2254.22,
    2264.32, 2274.42, 2284.52, 2294.61, 2304.71, 2314.81, 2324.91, 2335.01,
    2345.11, 2355.21
]  # Total: 46+16+18+31+34 = 145 bands

# EnMAP VNIR (First 91 bands)
HYSPECNET_VNIR_RAW = [
    418.42, 424.04, 429.46, 434.69, 439.76, 444.70, 449.54, 454.31, 459.03, 463.73,
    468.41, 473.08, 477.74, 482.41, 487.09, 491.78, 496.50, 501.24, 506.02, 510.83,
    515.67, 520.55, 525.47, 530.42, 535.42, 540.46, 545.55, 550.69, 555.87, 561.11,
    566.40, 571.76, 577.17, 582.64, 588.17, 593.77, 599.45, 605.19, 611.02, 616.92,
    622.92, 628.99, 635.11, 641.29, 647.54, 653.84, 660.21, 666.64, 673.13, 679.69,
    686.32, 693.01, 699.78, 706.62, 713.52, 720.50, 727.54, 734.65, 741.83, 749.06,
    756.35, 763.70, 771.11, 778.57, 786.08, 793.64, 801.25, 808.90, 816.61, 824.36,
    832.14, 839.98, 847.85, 855.76, 863.70, 871.68, 879.69, 887.73, 895.79, 903.87,
    911.97, 920.08, 928.20, 936.34, 944.47, 952.61, 960.75, 968.89, 977.04, 985.19,
    993.34
]

# EnMAP SWIR (Next 133 bands) - Provided by User
HYSPECNET_SWIR_RAW = [
    901.96, 911.57, 921.32, 931.20, 941.22, 951.36, 961.63, 972.02, 982.52, 993.14,
    1003.88, 1014.72, 1025.66, 1036.70, 1047.84, 1059.07, 1070.39, 1081.78, 1093.26, 1104.81,
    1116.43, 1128.10, 1139.84, 1151.62, 1163.44, 1175.30, 1187.20, 1199.11, 1211.05, 1223.00,
    1234.97, 1246.94, 1258.93, 1270.92, 1282.92, 1294.91, 1306.90, 1318.88, 1330.85, 1342.82,
    1354.76, 1366.69, 1378.60, 1390.48, 1461.10, 1472.74, 1484.34, 1495.89, 1507.40, 1518.87,
    1530.29, 1541.67, 1553.01, 1564.30, 1575.55, 1586.76, 1597.91, 1609.02, 1620.09, 1631.11,
    1642.07, 1653.00, 1663.87, 1674.70, 1685.47, 1696.20, 1706.87, 1717.50, 1728.08, 1738.60,
    1749.08, 1759.51, 1939.14, 1948.69, 1958.20, 1967.66, 1977.08, 1986.45, 1995.79, 2005.08,
    2014.33, 2023.54, 2032.70, 2041.83, 2050.92, 2059.96, 2068.97, 2077.93, 2086.86, 2095.74,
    2104.59, 2113.40, 2122.17, 2130.90, 2139.60, 2148.26, 2156.88, 2165.47, 2174.02, 2182.53,
    2191.01, 2199.45, 2207.86, 2216.24, 2224.58, 2232.89, 2241.16, 2249.40, 2257.61, 2265.79,
    2273.93, 2282.04, 2290.12, 2298.17, 2306.19, 2314.17, 2322.13, 2330.05, 2337.94, 2345.81,
    2353.64, 2361.44, 2369.21, 2376.95, 2384.66, 2392.34, 2400.00, 2407.62, 2415.21, 2422.78,
    2430.32, 2437.82, 2445.30
]

# EnMAP full 224-band center wavelengths from EnMAP_Spectral_Bands_update.xlsx (VNIR + SWIR)
HYSPECNET224_FULL_WAVELENGTHS = [
    418.416, 424.043, 429.457, 434.686, 439.758, 444.699, 449.539, 454.306, 459.031, 463.730,
    468.411, 473.080, 477.744, 482.411, 487.087, 491.780, 496.497, 501.243, 506.020, 510.829,
    515.672, 520.551, 525.467, 530.424, 535.422, 540.463, 545.551, 550.687, 555.873, 561.112,
    566.405, 571.756, 577.166, 582.636, 588.171, 593.773, 599.446, 605.193, 611.017, 616.923,
    622.921, 628.987, 635.112, 641.294, 647.537, 653.841, 660.207, 666.637, 673.131, 679.691,
    686.319, 693.014, 699.780, 706.617, 713.524, 720.501, 727.545, 734.654, 741.826, 749.060,
    756.353, 763.703, 771.108, 778.567, 786.078, 793.639, 801.249, 808.905, 816.608, 824.355,
    832.145, 839.976, 847.847, 855.757, 863.703, 871.683, 879.693, 887.729, 895.789, 903.870,
    911.968, 920.081, 928.204, 936.335, 944.470, 952.608, 960.748, 968.892, 977.037, 985.186,
    993.338, 901.961, 911.571, 921.320, 931.203, 941.218, 951.360, 961.628, 972.016, 982.523,
    993.144, 1003.880, 1014.720, 1025.660, 1036.700, 1047.840, 1059.070, 1070.390, 1081.780, 1093.260,
    1104.810, 1116.430, 1128.100, 1139.840, 1151.620, 1163.440, 1175.300, 1187.200, 1199.110, 1211.050,
    1223.000, 1234.970, 1246.940, 1258.930, 1270.920, 1282.920, 1294.910, 1306.900, 1318.880, 1330.850,
    1342.820, 1354.760, 1366.690, 1378.600, 1390.480, 1461.100, 1472.740, 1484.340, 1495.890, 1507.400,
    1518.870, 1530.290, 1541.670, 1553.010, 1564.300, 1575.550, 1586.760, 1597.910, 1609.020, 1620.090,
    1631.110, 1642.070, 1653.000, 1663.870, 1674.700, 1685.470, 1696.200, 1706.870, 1717.500, 1728.080,
    1738.600, 1749.080, 1759.510, 1939.140, 1948.690, 1958.200, 1967.660, 1977.080, 1986.450, 1995.790,
    2005.080, 2014.330, 2023.540, 2032.700, 2041.830, 2050.920, 2059.960, 2068.970, 2077.930, 2086.860,
    2095.740, 2104.590, 2113.400, 2122.170, 2130.900, 2139.600, 2148.260, 2156.880, 2165.470, 2174.020,
    2182.530, 2191.010, 2199.450, 2207.860, 2216.240, 2224.580, 2232.890, 2241.160, 2249.400, 2257.610,
    2265.790, 2273.930, 2282.040, 2290.120, 2298.170, 2306.190, 2314.170, 2322.130, 2330.050, 2337.940,
    2345.810, 2353.640, 2361.440, 2369.210, 2376.950, 2384.660, 2392.340, 2400.000, 2407.620, 2415.210,
    2422.780, 2430.320, 2437.820, 2445.300
]

# Precision NEON (369 Bands) - Filtered from 426 bands (Removed bad bands)
NEON_WAVELENGTHS = [
    382.2587, 387.2674, 392.2761, 397.2848, 402.2935, 407.3022, 412.3109, 417.3195, 422.3282, 427.3369,
    432.3455, 437.3542, 442.3630, 447.3716, 452.3804, 457.3890, 462.3978, 467.4064, 472.4150, 477.4238,
    482.4324, 487.4412, 492.4498, 497.4585, 502.4672, 507.4759, 512.4846, 517.4932, 522.5020, 527.5106,
    532.5194, 537.5280, 542.5367, 547.5454, 552.5542, 557.5628, 562.5714, 567.5802, 572.5888, 577.5974,
    582.6062, 587.6148, 592.6236, 597.6322, 602.6410, 607.6496, 612.6584, 617.6670, 622.6757, 627.6844,
    632.6930, 637.7017, 642.7104, 647.7191, 652.7278, 657.7365, 662.7452, 667.7538, 672.7626, 677.7712,
    682.7800, 687.7886, 692.7974, 697.8060, 702.8148, 707.8233, 712.8320, 717.8408, 722.8495, 727.8580,
    732.8668, 737.8754, 742.8842, 747.8928, 752.9016, 757.9102, 762.9189, 767.9276, 772.9364, 777.9450,
    782.9538, 787.9624, 792.9712, 797.9798, 802.9884, 807.9970, 813.0058, 818.0144, 823.0230, 828.0318,
    833.0404, 838.0492, 843.0579, 848.0665, 853.0753, 858.0840, 863.0927, 868.1014, 873.1100, 878.1187,
    883.1274, 888.1361, 893.1447, 898.1534, 903.1621, 908.1708, 913.1795, 918.1882, 923.1968, 928.2056,
    933.2143, 938.2230, 943.2316, 948.2403, 953.2490, 958.2578, 963.2663, 968.2750, 973.2838, 978.2924,
    983.3010, 988.3098, 993.3184, 998.3272, 1003.3359, 1008.3444, 1013.3532, 1018.3619, 1023.3706, 1028.3793,
    1033.3880, 1038.3966, 1043.4054, 1048.4140, 1053.4227, 1058.4314, 1063.4401, 1068.4487, 1073.4574, 1078.4661,
    1083.4749, 1088.4836, 1093.4922, 1098.5009, 1103.5095, 1108.5182, 1113.5270, 1118.5356, 1123.5443, 1128.5530,
    1133.5616, 1138.5704, 1143.5791, 1148.5878, 1153.5964, 1158.6051, 1163.6138, 1168.6224, 1173.6312, 1178.6398,
    1183.6484, 1188.6571, 1193.6658, 1198.6746, 1203.6833, 1208.6920, 1213.7006, 1218.7092, 1223.7180, 1228.7267,
    1233.7355, 1238.7441, 1243.7528, 1248.7616, 1253.7703, 1258.7789, 1263.7875, 1268.7963, 1273.8049, 1278.8136,
    1283.8223, 1288.8310, 1293.8397, 1298.8485, 1303.8570, 1308.8656, 1313.8743, 1318.8829, 1323.8917, 1328.9004,
    1333.9090, 1338.9178, 1449.1088, 1454.1174, 1459.1263, 1464.1350, 1469.1436, 1474.1522, 1479.1610, 1484.1697,
    1489.1785, 1494.1871, 1499.1959, 1504.2045, 1509.2135, 1514.2219, 1519.2306, 1524.2394, 1529.2479, 1534.2566,
    1539.2654, 1544.2742, 1549.2828, 1554.2914, 1559.3000, 1564.3088, 1569.3175, 1574.3260, 1579.3348, 1584.3436,
    1589.3522, 1594.3610, 1599.3695, 1604.3783, 1609.3870, 1614.3956, 1619.4043, 1624.4130, 1629.4215, 1634.4303,
    1639.4388, 1644.4475, 1649.4563, 1654.4650, 1659.4736, 1664.4824, 1669.4911, 1674.5000, 1679.5085, 1684.5175,
    1689.5260, 1694.5345, 1699.5432, 1704.5518, 1709.5605, 1714.5692, 1719.5780, 1724.5868, 1729.5955, 1734.6042,
    1739.6130, 1744.6215, 1749.6302, 1754.6388, 1759.6476, 1764.6562, 1769.6650, 1774.6738, 1779.6824, 1784.6910,
    1789.6996, 1959.9950, 1965.0035, 1970.0125, 1975.0208, 1980.0295, 1985.0385, 1990.0470, 1995.0558, 2000.0645,
    2005.0732, 2010.0820, 2015.0906, 2020.0992, 2025.1078, 2030.1165, 2035.1252, 2040.1340, 2045.1428, 2050.1516,
    2055.1600, 2060.1687, 2065.1772, 2070.1860, 2075.1948, 2080.2034, 2085.2122, 2090.2210, 2095.2295, 2100.2383,
    2105.2466, 2110.2556, 2115.2642, 2120.2730, 2125.2817, 2130.2903, 2135.2990, 2140.3079, 2145.3164, 2150.3250,
    2155.3337, 2160.3423, 2165.3510, 2170.3599, 2175.3684, 2180.3774, 2185.3860, 2190.3948, 2195.4033, 2200.4119,
    2205.4207, 2210.4292, 2215.4377, 2220.4468, 2225.4556, 2230.4640, 2235.4730, 2240.4817, 2245.4900, 2250.4988,
    2255.5073, 2260.5160, 2265.5247, 2270.5334, 2275.5422, 2280.5510, 2285.5598, 2290.5684, 2295.5770, 2300.5857,
    2305.5942, 2310.6030, 2315.6116, 2320.6204, 2325.6292, 2330.6380, 2335.6465, 2340.6550, 2345.6638, 2350.6724,
    2355.6812, 2360.6900, 2365.6985, 2370.7073, 2375.7160, 2380.7246, 2385.7332, 2390.7422, 2395.7507, 2400.7593,
    2405.7678, 2410.7769, 2415.7854, 2420.7942, 2425.8030, 2430.8113, 2435.8200, 2440.8286, 2445.8374, 2450.8460,
    2455.8550, 2460.8635, 2465.8723, 2470.8810, 2475.8900, 2480.8987, 2485.9070, 2490.9158, 2495.9243
]

# HyMap Karlsruhe (125 bands)
# Filtered based on meta data
KARLSRUHE_WAVELENGTHS = [
    454.2000, 466.9000, 482.9000, 497.4000, 512.5000, 528.4000, 543.8000, 558.9000, 574.0000, 589.2000, 
    604.5000, 619.9000, 635.4000, 650.6000, 665.7000, 681.0000, 696.5000, 711.8000, 726.9000, 742.0000, 
    757.3000, 772.4000, 787.3000, 802.6000, 817.9000, 833.1000, 848.2000, 863.6000, 878.8000, 893.2000, 
    891.0000, 907.0000, 923.4000, 939.3000, 955.1000, 971.0000, 986.6000, 1002.3000, 1018.0000, 1033.4000, 
    1048.8000, 1064.1000, 1079.4000, 1094.3000, 1109.2000, 1124.2000, 1139.0000, 1153.4000, 1167.9000, 1182.7000, 
    1197.1000, 1211.2000, 1225.4000, 1239.7000, 1253.9000, 1267.9000, 1282.0000, 1295.9000, 1309.7000, 1323.7000, 
    1337.5000, 1415.5000, 1429.6000, 1443.9000, 1458.0000, 1471.9000, 1486.0000, 1499.8000, 1513.3000, 1526.8000, 
    1540.5000, 1554.1000, 1567.3000, 1580.4000, 1593.6000, 1606.6000, 1619.5000, 1632.4000, 1645.4000, 1658.1000, 
    1670.6000, 1683.2000, 1695.9000, 1708.2000, 1720.4000, 1732.8000, 1745.2000, 1757.3000, 1769.2000, 1781.3000, 
    1793.3000, 1805.0000, 1816.6000, 1952.8000, 1972.0000, 1990.9000, 2009.9000, 2028.6000, 2047.1000, 2065.7000, 
    2084.0000, 2102.0000, 2120.0000, 2137.8000, 2155.5000, 2172.7000, 2189.5000, 2207.4000, 2225.3000, 2242.4000, 
    2259.5000, 2276.0000, 2293.2000, 2309.9000, 2326.1000, 2342.3000, 2359.4000, 2374.8000, 2390.7000, 2406.6000, 
    2422.4000, 2438.3000, 2453.8000, 2469.6000, 2484.6000
]

# CoCoaSpec (204 bands)
COCOASPEC_WAVELENGTHS = [
    397.32, 400.20, 403.09, 405.97, 408.85, 411.74, 414.63, 417.52, 420.40, 423.29,
    426.19, 429.08, 431.97, 434.87, 437.76, 440.66, 443.56, 446.45, 449.35, 452.25,
    455.16, 458.06, 460.96, 463.87, 466.77, 469.68, 472.59, 475.50, 478.41, 481.32,
    484.23, 487.14, 490.06, 492.97, 495.89, 498.80, 501.72, 504.64, 507.56, 510.48,
    513.40, 516.33, 519.25, 522.18, 525.10, 528.03, 530.96, 533.89, 536.82, 539.75,
    542.68, 545.62, 548.55, 551.49, 554.43, 557.36, 560.30, 563.24, 566.18, 569.12,
    572.07, 575.01, 577.96, 580.90, 583.85, 586.80, 589.75, 592.70, 595.65, 598.60,
    601.55, 604.51, 607.46, 610.42, 613.38, 616.34, 619.30, 622.26, 625.22, 628.18,
    631.15, 634.11, 637.08, 640.04, 643.01, 645.98, 648.95, 651.92, 654.89, 657.87,
    660.84, 663.81, 666.79, 669.77, 672.75, 675.73, 678.71, 681.69, 684.67, 687.65,
    690.64, 693.62, 696.61, 699.60, 702.58, 705.57, 708.57, 711.56, 714.55, 717.54,
    720.54, 723.53, 726.53, 729.53, 732.53, 735.53, 738.53, 741.53, 744.53, 747.54,
    750.54, 753.55, 756.56, 759.56, 762.57, 765.58, 768.60, 771.61, 774.62, 777.64,
    780.65, 783.67, 786.68, 789.70, 792.72, 795.74, 798.77, 801.79, 804.81, 807.84,
    810.86, 813.89, 816.92, 819.95, 822.98, 826.01, 829.04, 832.07, 835.11, 838.14,
    841.18, 844.22, 847.25, 850.29, 853.33, 856.37, 859.42, 862.46, 865.50, 868.55,
    871.60, 874.64, 877.69, 880.74, 883.79, 886.84, 889.90, 892.95, 896.01, 899.06,
    902.12, 905.18, 908.24, 911.30, 914.36, 917.42, 920.48, 923.55, 926.61, 929.68,
    932.74, 935.81, 938.88, 941.95, 945.02, 948.10, 951.17, 954.24, 957.32, 960.40,
    963.47, 966.55, 969.63, 972.71, 975.79, 978.88, 981.96, 985.05, 988.13, 991.22,
    994.31, 997.40, 1000.49, 1003.58
]

# Cabbage-Eggplant (277 bands) from ENVI header:
# range 399.856995-949.599976 nm, shape (800, 4502, 277).
# The header wavelengths are near-uniform; using a fixed 277-point grid keeps
# channel count strict while remaining stable across files.
CABBAGE_EGGPLANT_WAVELENGTHS = np.linspace(399.856995, 949.599976, 277, dtype=np.float32)

# HyperspectralBlueberries (462 bands) from ENVI header:
# range 393.8-1010.6 nm, BIL, data type=12.
HYPERSPECTRAL_BLUEBERRIES_WAVELENGTHS = np.linspace(393.8, 1010.6, 462, dtype=np.float32)

# PotatoWaterStress (448 bands) from ENVI header:
# metadata contains two wavelength segments in one list; keep the same 160+288 layout.
POTATO_WATER_STRESS_WAVELENGTHS = np.concatenate([
    np.linspace(409.759827, 988.258730, 160, dtype=np.float32),
    np.linspace(950.108239, 2509.552726, 288, dtype=np.float32),
]).astype(np.float32)

# SpectroFood Mushroom (204 bands) from user-provided sensor wavelengths.
SPECTROFOOD_MUSHROOM_WAVELENGTHS = [
    400.00, 402.95, 405.89, 408.84, 411.78, 414.73, 417.68, 420.62, 423.57, 426.51, 429.46, 432.41,
    435.35, 438.30, 441.24, 444.19, 447.14, 450.08, 453.03, 455.97, 458.92, 461.87, 464.81, 467.76,
    470.70, 473.65, 476.60, 479.54, 482.49, 485.43, 488.38, 491.33, 494.27, 497.22, 500.16, 503.11,
    506.06, 509.00, 511.95, 514.89, 517.84, 520.79, 523.73, 526.68, 529.62, 532.57, 535.52, 538.46,
    541.41, 544.35, 547.30, 550.25, 553.19, 556.14, 559.08, 562.03, 564.98, 567.92, 570.87, 573.81,
    576.76, 579.71, 582.65, 585.60, 588.54, 591.49, 594.44, 597.38, 600.33, 603.27, 606.22, 609.17,
    612.11, 615.06, 618.00, 620.95, 623.90, 626.84, 629.79, 632.73, 635.68, 638.63, 641.57, 644.52,
    647.46, 650.41, 653.36, 656.30, 659.25, 662.19, 665.14, 668.09, 671.03, 673.98, 676.92, 679.87,
    682.82, 685.76, 688.71, 691.65, 694.60, 697.55, 700.49, 703.44, 706.38, 709.33, 712.28, 715.22,
    718.17, 721.11, 724.06, 727.01, 729.95, 732.90, 735.84, 738.79, 741.74, 744.68, 747.63, 750.57,
    753.52, 756.47, 759.41, 762.36, 765.30, 768.25, 771.20, 774.14, 777.09, 780.03, 782.98, 785.93,
    788.87, 791.82, 794.76, 797.71, 800.66, 803.60, 806.55, 809.49, 812.44, 815.39, 818.33, 821.28,
    824.22, 827.17, 830.12, 833.06, 836.01, 838.95, 841.90, 844.85, 847.79, 850.74, 853.68, 856.63,
    859.58, 862.52, 865.47, 868.41, 871.36, 874.31, 877.25, 880.20, 883.14, 886.09, 889.04, 891.98,
    894.93, 897.87, 900.82, 903.77, 906.71, 909.66, 912.60, 915.55, 918.50, 921.44, 924.39, 927.33,
    930.28, 933.23, 936.17, 939.12, 942.06, 945.01, 947.96, 950.90, 953.85, 956.79, 959.74, 962.69,
    965.63, 968.58, 971.52, 974.47, 977.42, 980.36, 983.31, 986.25, 989.20, 992.15, 995.09, 998.04
]

# SpectroFood Apple (141 bands, 430-990nm, 4nm spacing).
SPECTROFOOD_APPLE_WAVELENGTHS = np.linspace(430.0, 990.0, 141, dtype=np.float32)

# SpectroFood Broccoli (150 bands, 470-900nm).
SPECTROFOOD_BROCCOLI_WAVELENGTHS = np.linspace(470.0, 900.0, 150, dtype=np.float32)

# SpectroFood FX10 (224 bands) from user-provided sensor wavelengths.
SPECTROFOOD_FX10_WAVELENGTHS = [
    397.66, 400.28, 402.90, 405.52, 408.13, 410.75, 413.37, 416.00, 418.62, 421.24, 423.86, 426.49,
    429.12, 431.74, 434.37, 437.00, 439.63, 442.26, 444.89, 447.52, 450.16, 452.79, 455.43, 458.06,
    460.70, 463.34, 465.98, 468.62, 471.26, 473.90, 476.54, 479.18, 481.83, 484.47, 487.12, 489.77,
    492.42, 495.07, 497.72, 500.37, 503.02, 505.67, 508.32, 510.98, 513.63, 516.29, 518.95, 521.61,
    524.27, 526.93, 529.59, 532.25, 534.91, 537.57, 540.24, 542.91, 545.57, 548.24, 550.91, 553.58,
    556.25, 558.92, 561.59, 564.26, 566.94, 569.61, 572.29, 574.96, 577.64, 580.32, 583.00, 585.68,
    588.36, 591.04, 593.73, 596.41, 599.10, 601.78, 604.47, 607.16, 609.85, 612.53, 615.23, 617.92,
    620.61, 623.30, 626.00, 628.69, 631.39, 634.08, 636.78, 639.48, 642.18, 644.88, 647.58, 650.29,
    652.99, 655.69, 658.40, 661.10, 663.81, 666.52, 669.23, 671.94, 674.65, 677.36, 680.07, 682.79,
    685.50, 688.22, 690.93, 693.65, 696.37, 699.09, 701.81, 704.53, 707.25, 709.97, 712.70, 715.42,
    718.15, 720.87, 723.60, 726.33, 729.06, 731.79, 734.52, 737.25, 739.98, 742.72, 745.45, 748.19,
    750.93, 753.66, 756.40, 759.14, 761.88, 764.62, 767.36, 770.11, 772.85, 775.60, 778.34, 781.09,
    783.84, 786.58, 789.33, 792.08, 794.84, 797.59, 800.34, 803.10, 805.85, 808.61, 811.36, 814.12,
    816.88, 819.64, 822.40, 825.16, 827.92, 830.69, 833.45, 836.22, 838.98, 841.75, 844.52, 847.29,
    850.06, 852.83, 855.60, 858.37, 861.14, 863.92, 866.69, 869.47, 872.25, 875.03, 877.80, 880.58,
    883.37, 886.15, 888.93, 891.71, 894.50, 897.28, 900.07, 902.86, 905.64, 908.43, 911.22, 914.02,
    916.81, 919.60, 922.39, 925.19, 927.98, 930.78, 933.58, 936.38, 939.18, 941.98, 944.78, 947.58,
    950.38, 953.19, 955.99, 958.80, 961.60, 964.41, 967.22, 970.03, 972.84, 975.65, 978.46, 981.27,
    984.09, 986.90, 989.72, 992.54, 995.35, 998.17, 1000.99, 1003.81
]

# Agriculture-Forest (370 bands)
AG_FOREST_WAVELENGTHS = [
    381.8700, 386.8800, 391.8900, 396.8900, 401.9000, 406.9100, 411.9200, 416.9300, 421.9400, 426.9500, 
    431.9600, 436.9600, 441.9700, 446.9800, 451.9900, 457.0000, 462.0100, 467.0200, 472.0200, 477.0300, 
    482.0400, 487.0500, 492.0600, 497.0700, 502.0800, 507.0900, 512.0900, 517.1000, 522.1100, 527.1200, 
    532.1300, 537.1400, 542.1500, 547.1500, 552.1600, 557.1700, 562.1800, 567.1900, 572.2000, 577.2100, 
    582.2200, 587.2200, 592.2300, 597.2400, 602.2500, 607.2600, 612.2700, 617.2800, 622.2800, 627.2900, 
    632.3000, 637.3100, 642.3200, 647.3300, 652.3400, 657.3500, 662.3500, 667.3600, 672.3700, 677.3800, 
    682.3900, 687.4000, 692.4100, 697.4100, 702.4200, 707.4300, 712.4400, 717.4500, 722.4600, 727.4700, 
    732.4800, 737.4800, 742.4900, 747.5000, 752.5100, 757.5200, 762.5300, 767.5400, 772.5400, 777.5500, 
    782.5600, 787.5700, 792.5800, 797.5900, 802.6000, 807.6100, 812.6100, 817.6200, 822.6300, 827.6400, 
    832.6500, 837.6600, 842.6700, 847.6700, 852.6800, 857.6900, 862.7000, 867.7100, 872.7200, 877.7300, 
    882.7400, 887.7400, 892.7500, 897.7600, 902.7700, 907.7800, 912.7900, 917.8000, 922.8100, 927.8100, 
    932.8200, 937.8300, 942.8400, 947.8500, 952.8600, 957.8700, 962.8700, 967.8800, 972.8900, 977.9000, 
    982.9100, 987.9200, 992.9300, 997.9400, 1002.9400, 1007.9500, 1012.9600, 1017.9700, 1022.9800, 1027.9900, 
    1033.0000, 1038.0000, 1043.0100, 1048.0200, 1053.0300, 1058.0400, 1063.0500, 1068.0601, 1073.0699, 1078.0699, 
    1083.0800, 1088.0900, 1093.1000, 1098.1100, 1103.1200, 1108.1300, 1113.1300, 1118.1400, 1123.1500, 1128.1600, 
    1133.1700, 1138.1801, 1143.1899, 1148.2000, 1153.2000, 1158.2100, 1163.2200, 1168.2300, 1173.2400, 1178.2500, 
    1183.2600, 1188.2600, 1193.2700, 1198.2800, 1203.2900, 1208.3000, 1213.3101, 1218.3199, 1223.3300, 1228.3300, 
    1233.3400, 1238.3500, 1243.3600, 1248.3700, 1253.3800, 1258.3900, 1263.3900, 1268.4000, 1273.4100, 1278.4200, 
    1283.4301, 1288.4399, 1293.4500, 1298.4600, 1303.4600, 1308.4700, 1313.4800, 1318.4900, 1323.5000, 1328.5100, 
    1333.5200, 1338.5200, 1343.5300, 1433.6899, 1438.7000, 1443.7100, 1448.7200, 1453.7200, 1458.7300, 1463.7400, 
    1468.7500, 1473.7600, 1478.7700, 1483.7800, 1488.7800, 1493.7900, 1498.8000, 1503.8101, 1508.8199, 1513.8300, 
    1518.8400, 1523.8500, 1528.8500, 1533.8600, 1538.8700, 1543.8800, 1548.8900, 1553.9000, 1558.9100, 1563.9100, 
    1568.9200, 1573.9301, 1578.9399, 1583.9500, 1588.9600, 1593.9700, 1598.9800, 1603.9800, 1608.9900, 1614.0000, 
    1619.0100, 1624.0200, 1629.0300, 1634.0400, 1639.0400, 1644.0500, 1649.0601, 1654.0699, 1659.0800, 1664.0900, 
    1669.1000, 1674.1100, 1679.1100, 1684.1200, 1689.1300, 1694.1400, 1699.1500, 1704.1600, 1709.1700, 1714.1700, 
    1719.1801, 1724.1899, 1729.2000, 1734.2100, 1739.2200, 1744.2300, 1749.2400, 1754.2400, 1759.2500, 1764.2600, 
    1769.2700, 1774.2800, 1954.5900, 1959.6000, 1964.6100, 1969.6200, 1974.6300, 1979.6300, 1984.6400, 1989.6500, 
    1994.6600, 1999.6700, 2004.6801, 2009.6899, 2014.7000, 2019.7000, 2024.7100, 2029.7200, 2034.7300, 2039.7400, 
    2044.7500, 2049.7600, 2054.7600, 2059.7700, 2064.7800, 2069.7900, 2074.8000, 2079.8101, 2084.8201, 2089.8301, 
    2094.8301, 2099.8401, 2104.8501, 2109.8601, 2114.8701, 2119.8799, 2124.8899, 2129.8899, 2134.8999, 2139.9099, 
    2144.9199, 2149.9299, 2154.9399, 2159.9500, 2164.9600, 2169.9600, 2174.9700, 2179.9800, 2184.9900, 2190.0000, 
    2195.0100, 2200.0200, 2205.0200, 2210.0300, 2215.0400, 2220.0500, 2225.0601, 2230.0701, 2235.0801, 2240.0901, 
    2245.0901, 2250.1001, 2255.1101, 2260.1201, 2265.1299, 2270.1399, 2275.1499, 2280.1499, 2285.1599, 2290.1699, 
    2295.1799, 2300.1899, 2305.2000, 2310.2100, 2315.2200, 2320.2200, 2325.2300, 2330.2400, 2335.2500, 2340.2600, 
    2345.2700, 2350.2800, 2355.2800, 2360.2900, 2365.3000, 2370.3101, 2375.3201, 2380.3301, 2385.3401, 2390.3501, 
    2395.3501, 2400.3601, 2405.3701, 2410.3799, 2415.3899, 2420.3999, 2425.4099, 2430.4099, 2435.4199, 2440.4299, 
    2445.4399, 2450.4500, 2455.4600, 2460.4700, 2465.4800, 2470.4800, 2475.4900, 2480.5000, 2485.5100, 2490.5200
]

# Samson (156 Bands) - Extracted from samson_1.img.hdr
SAMSON_WAVELENGTHS = [
    400.9387, 404.1525, 407.3663, 410.5800, 413.7939, 417.0076, 420.2214, 423.4352, 426.6490, 429.8627,
    433.0765, 436.2903, 439.5041, 442.7179, 445.9316, 449.1454, 452.3592, 455.5730, 458.7868, 462.0005,
    465.2143, 468.4281, 471.6419, 474.8557, 478.0694, 481.2832, 484.4970, 487.7108, 490.9246, 494.1383,
    497.3521, 500.5659, 503.7797, 506.9934, 510.2072, 513.4210, 516.6348, 519.8486, 523.0624, 526.2761,
    529.4899, 532.7037, 535.9175, 539.1312, 542.3450, 545.5588, 548.7726, 551.9864, 555.2001, 558.4139,
    561.6277, 564.8415, 568.0553, 571.2690, 574.4828, 577.6966, 580.9104, 584.1241, 587.3380, 590.5518,
    593.7655, 596.9793, 600.1931, 603.4069, 606.6206, 609.8344, 613.0482, 616.2620, 619.4758, 622.6895,
    625.9033, 629.1171, 632.3309, 635.5446, 638.7584, 641.9722, 645.1860, 648.3998, 651.6135, 654.8273,
    658.0411, 661.2549, 664.4687, 667.6824, 670.8962, 674.1100, 677.3238, 680.5375, 683.7513, 686.9651,
    690.1789, 693.3927, 696.6064, 699.8203, 703.0340, 706.2478, 709.4615, 712.6754, 715.8892, 719.1029,
    722.3167, 725.5305, 728.7443, 731.9580, 735.1718, 738.3856, 741.5994, 744.8132, 748.0269, 751.2407,
    754.4545, 757.6683, 760.8821, 764.0958, 767.3096, 770.5234, 773.7372, 776.9509, 780.1647, 783.3785,
    786.5923, 789.8060, 793.0198, 796.2336, 799.4474, 802.6612, 805.8749, 809.0887, 812.3025, 815.5163,
    818.7301, 821.9438, 825.1577, 828.3714, 831.5852, 834.7989, 838.0128, 841.2266, 844.4403, 847.6541,
    850.8679, 854.0817, 857.2954, 860.5092, 863.7230, 866.9368, 870.1506, 873.3643, 876.5781, 879.7919,
    883.0057, 886.2194, 889.4332, 892.6470, 895.8608, 899.0746,
]

# Washington DC Mall (191 Bands)
# Updated from 'washington_dc_mall wavelengths.txt'
DCMALL_WAVELENGTHS = [
    401.288, 404.590, 407.919, 411.279, 414.671, 418.100, 421.568, 425.078, 428.632, 432.235,
    435.888, 439.595, 443.358, 447.182, 451.067, 455.019, 459.040, 463.134, 467.302, 471.550,
    475.880, 480.298, 484.805, 489.406, 494.105, 498.906, 503.814, 508.832, 513.966, 519.221,
    524.600, 530.109, 535.752, 541.537, 547.467, 553.547, 559.784, 566.183, 572.751, 579.491,
    586.411, 593.518, 600.815, 608.309, 616.006, 623.912, 632.031, 640.368, 648.930, 657.721,
    666.745, 676.005, 685.506, 695.249, 705.237, 715.473, 725.956, 736.686, 747.664, 758.887,
    770.353, 782.060, 794.003, 806.177, 818.576, 831.195, 844.025, 857.059, 870.287, 883.702,
    897.292, 911.048, 924.959, 939.015, 953.203, 967.514, 981.935, 996.454, 1011.06, 1025.75,
    1040.51, 1055.31, 1070.16, 1085.06, 1099.97, 1114.90, 1129.85, 1144.79, 1159.73, 1174.65,
    1189.55, 1204.42, 1219.26, 1234.06, 1248.82, 1263.53, 1278.19, 1292.78, 1307.32, 1321.80,
    1336.21, 1350.55, 1421.14, 1435.03, 1448.84, 1462.57, 1476.21, 1489.78, 1503.26, 1516.67,
    1529.99, 1543.22, 1556.38, 1569.46, 1582.44, 1595.35, 1608.19, 1620.93, 1633.61, 1646.20,
    1658.70, 1671.14, 1683.49, 1695.77, 1707.97, 1720.10, 1732.15, 1744.12, 1756.03, 1767.86,
    1779.62, 1791.31, 1802.93, 1937.15, 1947.92, 1958.64, 1969.29, 1979.89, 1990.43, 2000.91,
    2011.34, 2021.71, 2032.02, 2042.28, 2052.49, 2062.64, 2072.74, 2082.79, 2092.79, 2102.74,
    2112.64, 2122.49, 2132.29, 2142.04, 2151.74, 2161.40, 2171.00, 2180.57, 2190.08, 2199.55,
    2208.98, 2218.36, 2227.70, 2237.00, 2246.26, 2255.47, 2264.63, 2273.76, 2282.84, 2291.89,
    2300.90, 2309.86, 2318.79, 2327.67, 2336.53, 2345.34, 2354.11, 2362.84, 2371.54, 2380.21,
    2388.83, 2397.42, 2405.97, 2414.49, 2422.98, 2431.42, 2439.84, 2448.22, 2456.57, 2464.88,
    2473.16
]

# Urban_F210 (210 Bands), parsed from URBAN.wvl
URBAN_WAVELENGTHS = [
    399.000, 402.000, 405.000, 408.000, 412.000, 415.000, 419.000, 422.000, 426.000, 429.000, 433.000, 437.000,
    440.000, 444.000, 448.000, 452.000, 456.000, 460.000, 464.000, 468.000, 472.000, 477.000, 481.000, 486.000,
    490.000, 495.000, 500.000, 505.000, 510.000, 515.000, 520.000, 526.000, 531.000, 537.000, 543.000, 548.000,
    555.000, 561.000, 567.000, 574.000, 581.000, 588.000, 595.000, 602.000, 610.000, 617.000, 625.000, 633.000,
    642.000, 650.000, 659.000, 668.000, 678.000, 687.000, 697.000, 707.000, 717.000, 728.000, 739.000, 750.000,
    761.000, 772.000, 784.000, 796.000, 808.000, 821.000, 833.000, 846.000, 859.000, 873.000, 886.000, 900.000,
    913.000, 927.000, 941.000, 956.000, 970.000, 984.000, 999.000, 1014.000, 1028.000, 1043.000, 1058.000, 1073.000,
    1088.000, 1102.000, 1117.000, 1132.000, 1147.000, 1162.000, 1177.000, 1192.000, 1207.000, 1222.000, 1237.000, 1251.000,
    1266.000, 1281.000, 1295.000, 1310.000, 1324.000, 1339.000, 1353.000, 1367.000, 1381.000, 1395.000, 1410.000, 1423.000,
    1437.000, 1451.000, 1465.000, 1478.000, 1492.000, 1506.000, 1519.000, 1532.000, 1545.000, 1559.000, 1572.000, 1585.000,
    1598.000, 1610.000, 1623.000, 1636.000, 1648.000, 1661.000, 1673.000, 1686.000, 1698.000, 1710.000, 1722.000, 1734.000,
    1746.000, 1758.000, 1770.000, 1782.000, 1793.000, 1805.000, 1816.000, 1828.000, 1839.000, 1851.000, 1862.000, 1873.000,
    1884.000, 1895.000, 1906.000, 1917.000, 1928.000, 1939.000, 1950.000, 1960.000, 1971.000, 1982.000, 1992.000, 2003.000,
    2013.000, 2023.000, 2034.000, 2044.000, 2054.000, 2064.000, 2074.000, 2084.000, 2094.000, 2104.000, 2114.000, 2124.000,
    2134.000, 2144.000, 2153.000, 2163.000, 2173.000, 2182.000, 2192.000, 2201.000, 2211.000, 2220.000, 2229.000, 2239.000,
    2248.000, 2257.000, 2266.000, 2275.000, 2284.000, 2293.000, 2302.000, 2311.000, 2320.000, 2329.000, 2338.000, 2347.000,
    2356.000, 2364.000, 2373.000, 2382.000, 2390.000, 2399.000, 2407.000, 2416.000, 2424.000, 2433.000, 2441.000, 2450.000,
    2458.000, 2466.000, 2475.000, 2483.000, 2491.000, 2499.000
]

# EnMAP VNIR (First ~90 bands relevant for 450-950nm). Using linear approximation 420-1000nm.
# HySpecNet uses 202 bands from 420-2450nm.
# We will define a generator in `get_source_wavelengths`.

# ================= Utils =================

def sinc(x):
    """Normalized sinc function: sin(pi * x) / (pi * x)"""
    return np.sinc(x)

def lanczos_kernel(x, a=3):
    """Lanczos kernel function."""
    mask = np.abs(x) < a
    return np.where(mask, sinc(x) * sinc(x / a), 0)

# Helper: Create temporary header for known missing headers (Samson)
def create_temp_samson_header(img_path):
    import shutil
    temp_dir = os.path.join(OUTPUT_DIR, 'temp_headers')
    os.makedirs(temp_dir, exist_ok=True)
    
    base_name = os.path.basename(img_path)
    temp_img_path = os.path.join(temp_dir, base_name)
    temp_hdr_path = os.path.splitext(temp_img_path)[0] + '.hdr'
    
    # 1. Copy image to temp (because input is read-only and ENVI expects .hdr next to .img)
    if not os.path.exists(temp_img_path):
        print(f"  > Copying {img_path} to {temp_img_path} for header fix...", flush=True)
        shutil.copy(img_path, temp_img_path)
        
    # 2. Write Header Content (from user notes)
    header_content = """ENVI
description = {
  File Resize Result, x resize factor: 1.000000, y resize factor: 1.000000.
  [Wed Aug 29 14:01:05 2012]}
samples = 95
lines   = 95
bands   = 156
header offset = 0
file type = ENVI Standard
data type = 2
interleave = bsq
sensor type = Unknown
byte order = 0
map info = {UTM, 1.000, 1.000, 612730.910, 4074488.708, 1.0000000000e+00, 1.0000000000e+00, 10, North, WGS-84, units=Meters}
coordinate system string = {PROJCS["UTM_Zone_10N",GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",637813..0,298.252223563]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]],PROJECTION["Transverse_Mercator"],PARAMETER["False_Easting",50000.00],PARAMETER["False_Northing",0.0],PARAMETER["Central_Meridian",-123.0],PARAMETER["Scale_Factor",0.9996],PARAMETER["Latitude_Of_Origin",0.0],UNIT["Meter",1.0]]}
default bands = {85,50,10}
wavelength units = Nanometers
wavelength = {
 400.938721, 404.152496, 407.366302, 410.580048, 413.793854, 417.007629,
 420.221405, 423.435181, 426.648956, 429.862732, 433.076508, 436.290314,
 439.504089, 442.717865, 445.931641, 449.145416, 452.359192, 455.572968,
 458.786774, 462.000549, 465.214325, 468.428101, 471.641876, 474.855652,
 478.069427, 481.283234, 484.497009, 487.710785, 490.924561, 494.138336,
 497.352112, 500.565887, 503.779694, 506.993439, 510.207245, 513.421021,
 516.634766, 519.848572, 523.062378, 526.276123, 529.489929, 532.703674,
 535.917480, 539.131226, 542.345032, 545.558838, 548.772583, 551.986389,
 555.200134, 558.413940, 561.627686, 564.841492, 568.055298, 571.269043,
 574.482849, 577.696594, 580.910400, 584.124146, 587.337952, 590.551758,
 593.765503, 596.979309, 600.193054, 603.406860, 606.620605, 609.834412,
 613.048157, 616.261963, 619.475769, 622.689514, 625.903320, 629.117065,
 632.330872, 635.544617, 638.758423, 641.972229, 645.185974, 648.399780,
 651.613525, 654.827332, 658.041077, 661.254883, 664.468689, 667.682434,
 670.896240, 674.109985, 677.323792, 680.537537, 683.751343, 686.965149,
 690.178894, 693.392700, 696.606445, 699.820251, 703.033997, 706.247803,
 709.461548, 712.675354, 715.889160, 719.102905, 722.316711, 725.530457,
 728.744263, 731.958008, 735.171814, 738.385620, 741.599365, 744.813171,
 748.026917, 751.240723, 754.454468, 757.668274, 760.882080, 764.095825,
 767.309631, 770.523376, 773.737183, 776.950928, 780.164734, 783.378540,
 786.592285, 789.806030, 793.019836, 796.233643, 799.447388, 802.661194,
 805.874939, 809.088745, 812.302490, 815.516296, 818.730103, 821.943848,
 825.157654, 828.371399, 831.585205, 834.798950, 838.012756, 841.226562,
 844.440308, 847.654114, 850.867859, 854.081665, 857.295410, 860.509216,
 863.723022, 866.936768, 870.150574, 873.364319, 876.578125, 879.791870,
 883.005676, 886.219421, 889.433228, 892.647034, 895.860779, 899.074585}
"""
    with open(temp_hdr_path, 'w') as f:
        f.write(header_content)
        
    return temp_img_path

def get_source_wavelengths(dataset_name):
    """
    Returns source wavelengths for various datasets.
    Based on HDR specs or official dataset documentation.
    """
    if dataset_name == 'HanChuan':
         return np.linspace(398.565, 1001.92, 274)
    elif dataset_name == 'HongHu':
         return np.linspace(401.810, 999.280, 270)
    elif dataset_name == 'LongKou':
         return np.linspace(401.810, 999.280, 270)
         
    # --- New Datasets Mappings ---
    # Based on user-provided 'Salinas鏁寸悊娉㈡.txt' and 'HyRANK鏁寸悊娉㈡.txt'
    
    elif dataset_name == 'Salinas':
        # Extracted from 'Salinas鏁寸悊娉㈡.txt' (224 bands)
        # The file lists bands 1-224, with wavelengths from ~360 nm to ~2510 nm.
        # We assume regular spacing for missing individual points or interpolate
        # But it's safer to reconstruct the list if it's irregular.
        # However, AVIRIS is typically 10nm spacing.
        # Using linspace to approximate based on Start/End from file logic:
        # Band 1 Center: ~360nm, Band 224 Center: ~2510nm
        return np.linspace(360, 2510, 224) 

    elif dataset_name == 'Indian_pines':
        # AVIRIS sensor minus 4 bands (220 total).
        # It's commonly treated as 400-2500nm span.
        return np.linspace(400, 2500, 220)

    elif dataset_name in ['Dioni', 'Loukia']:
        # HyRANK dataset (176 bands). 
        # Source: 'HyRANK鏁寸悊娉㈡.txt'
        # The list provided has specific center wavelengths.
        # Since it's irregular (due to water absorption removal), hardcoding or 
        # a more precise list is better.
        # Given the file provided has explicit values, we will use a simplified 
        # linear approximation for now, or we can paste the long list.
        # For robustness in this script, I'll use a linspace approximation 
        # for the active range 400-2400nm, which covers S185's 450-950nm well.
        # The S185 range (450-950) is fully within the first ~65 bands of HyRANK.
        return np.linspace(426.82, 2395.50, 176) # Start/End from txt file lines 15 & 606

    elif dataset_name == 'HyperLeaf2024':
         return np.array(HYPERLEAF_WAVELENGTHS)

    elif dataset_name == 'HyRANK':
        return np.array(HYRANK_WAVELENGTHS)
    
    elif dataset_name == 'Chikusei':
        return np.array(CHIKUSEI_WAVELENGTHS)

    elif dataset_name in ['KSC', 'Moffett']:
        # KSC (176 bands). Source: Salinas 224.
        # Moffett Field also uses this 176-band AVIRIS subset.
        # User Corrected Mask (Band Numbers 1-based -> Indices 0-based):
        # 1. Band 103-110 -> Index 102-109 (size 8).
        # 2. Band 149-175 -> Index 148-174 (size 27).
        # 3. Band 212-224 -> Index 211-223 (size 13).
        # Total removed: 48. Remaining: 176. Matches dataset exactly.
        
        full_224 = np.array(SALINAS_FULL_WAVELENGTHS)
        
        # 0-based indices
        r1 = list(range(102, 110))
        r2 = list(range(148, 175))
        r3 = list(range(211, 224))
        
        to_remove = r1 + r2 + r3
        ksc_wavs = np.delete(full_224, to_remove)
        
        if len(ksc_wavs) != 176:
             print(f"[WARNING] KSC Wavelengths count mismatch! Expected 176, got {len(ksc_wavs)}")

        return ksc_wavs

    elif dataset_name == 'HySpecNet224Ext':
        return np.array(HYSPECNET224_FULL_WAVELENGTHS)

    elif dataset_name == 'HySpecNet':
        # EnMAP 202 bands. Construction: VNIR + SWIR -> 224 bands -> Remove noisy bands.
        
        # 1. Concatenate Raw
        vnir = np.array(HYSPECNET_VNIR_RAW)
        swir = np.array(HYSPECNET_SWIR_RAW)
        full_224 = np.concatenate([vnir, swir])
        
        # 2. Remove Bands (0-based indices)
        kept_wavs = np.delete(full_224, HYSPECNET_WATER_VAPOR_REMOVE_INDICES)
        
        # Verify length
        if len(kept_wavs) != 202:
            print(f"[WARNING] HySpecNet Wavelengths count mismatch! Expected 202, got {len(kept_wavs)}. Raw sum: {len(full_224)}")
            
        return kept_wavs

    elif dataset_name == 'Kochia':
         # 224 Bands. Specim FX10 sensor: 400-1000nm
         return np.linspace(400, 1000, 224)

    elif dataset_name == 'SpectroFood_Apple':
        return np.array(SPECTROFOOD_APPLE_WAVELENGTHS)

    elif dataset_name == 'SpectroFood_Broccoli':
        return np.array(SPECTROFOOD_BROCCOLI_WAVELENGTHS)

    elif dataset_name == 'SpectroFood_Mushroom':
        return np.array(SPECTROFOOD_MUSHROOM_WAVELENGTHS)

    elif dataset_name == 'SpectroFood_FX10':
        return np.array(SPECTROFOOD_FX10_WAVELENGTHS)

    elif dataset_name == 'WeedHSI_Reflectance':
        # Specim IQ 204-band grid is identical to CoCoaSpec's wavelength list.
        return np.array(COCOASPEC_WAVELENGTHS)

    elif dataset_name == 'Cabbage_Eggplant':
        return np.array(CABBAGE_EGGPLANT_WAVELENGTHS)

    elif dataset_name == 'HyperspectralBlueberries':
        return np.array(HYPERSPECTRAL_BLUEBERRIES_WAVELENGTHS)

    elif dataset_name == 'PotatoWaterStress':
        return np.array(POTATO_WATER_STRESS_WAVELENGTHS)

    elif dataset_name == 'Botswana':
        # 145 Bands from Hyperion EO-1. Exact wavelengths (not approximation)
        return np.array(BOTSWANA_WAVELENGTHS)

    elif dataset_name == 'EarthView-Neon':
        # NEON (369 Bands). Source: filtered neon.txt (369 lines)
        return np.array(NEON_WAVELENGTHS)

    elif dataset_name == 'Cuprite':
        # Reuse Salinas (AVIRIS 224)
        return np.linspace(360, 2510, 224) 

    elif dataset_name == 'Samson':
        # 156 Bands
        return np.array(SAMSON_WAVELENGTHS)

    elif dataset_name == 'Urban':
        return np.array(URBAN_WAVELENGTHS)

    elif dataset_name == 'WashingtonDC':
        # Washington DC Mall uses the curated 191-band list.
        return np.array(DCMALL_WAVELENGTHS)

    elif dataset_name == 'Karlsruhe':
        return np.array(KARLSRUHE_WAVELENGTHS)
    
    elif dataset_name == 'CoCoaSpec':
        return np.array(COCOASPEC_WAVELENGTHS)

    elif dataset_name == 'AgForest':
        return np.array(AG_FOREST_WAVELENGTHS)

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def ensure_wavelength_count(source_wavelengths, band_count, dataset_name):
    """
    Ensure wavelength count matches the loaded cube's channel count.
    If mismatch happens (metadata/header inconsistency), interpolate a fallback grid.
    """
    source_wavelengths = np.asarray(source_wavelengths, dtype=np.float32)
    if len(source_wavelengths) == band_count:
        return source_wavelengths

    mismatch_key = (dataset_name, int(len(source_wavelengths)), int(band_count))
    with PRINT_LOCK:
        if mismatch_key not in PRINTED_WAVELENGTH_MISMATCHES:
            print(
                f"  [WARNING] Wavelength count mismatch for {dataset_name}: "
                f"{len(source_wavelengths)} (metadata) vs {band_count} (image). "
                f"Interpolating fallback wavelengths.",
                flush=True
            )
            PRINTED_WAVELENGTH_MISMATCHES.add(mismatch_key)
    return np.linspace(float(source_wavelengths[0]), float(source_wavelengths[-1]), band_count).astype(np.float32)

def lanczos_kernel_torch(x, a=3):
    """Lanczos kernel function (PyTorch version)."""
    # x is a tensor
    mask = torch.abs(x) < a
    # pi * x
    pix = math.pi * x
    # sinc(x) = sin(pi*x) / (pi*x)
    # PyTorch sinc is sin(pi*x)/(pi*x) already? 
    # torch.sinc(x) computes sin(pi*x)/(pi*x). Correct.
    return torch.where(mask, torch.sinc(x) * torch.sinc(x / a), torch.tensor(0.0, device=x.device))

def lanczos_resample_gpu(image_tensor, source_wavelengths, target_wavelengths, a=3):
    """
    GPU-accelerated Lanczos resampling using PyTorch.
    image_tensor: (H, W, C_in) float32 tensor on CUDA
    """
    H, W, C = image_tensor.shape
    C_out = len(target_wavelengths)
    device = image_tensor.device
    
    # Flatten image to (N, C)
    flat_image = image_tensor.reshape(-1, C) # (N, C_in)
    
    # We need a transformation matrix M of shape (C_in, C_out)
    # resampled = flat_image @ M
    # But Lanczos weights depend on target_wl relative to source_wls.
    # W_ij = Weight of source_band i contributing to target_band j.
    
    # Pre-calculate weights on CPU or GPU?
    # Source/Target lists are small (hundreds), quick to do on CPU then move to GPU.
    
    # Construct Weight Matrix (C_in, C_out)
    # BUT Lanczos is a convolution-like operation in spectral domain.
    # For each target_wl, we find source indices within window 'a'.
    
    # Let's build the sparse weight matrix.
    # Rows: Source indices (0..C-1)
    # Cols: Target indices (0..C_out-1)
    # It's actually easier to compute (C_out, C_in) for matmul?
    # Resampled = Weights (C_out, C_in) @ Image^T (C_in, N) -> (C_out, N) -> Transpose back
    # OR: Image (N, C_in) @ Weights^T (C_in, C_out)
    
    weight_matrix = torch.zeros((C, C_out), device=device, dtype=torch.float32)
    
    source_indices = np.arange(len(source_wavelengths))
    
    # Vectorized calculation might be tricky due to dynamic window, loop is fine for 125 bands.
    # print(f"  [GPU] Computing Lanczos weights for {C_out} target bands...", flush=True)
    
    for j, target_wl in enumerate(target_wavelengths):
        # if j % 20 == 0:
        #     print(f"    ...Band {j}/{C_out}", flush=True)
        # Find float index
        float_idx = np.interp(target_wl, source_wavelengths, source_indices)
        center = math.floor(float_idx)
        start_idx = int(center - a + 1)
        end_idx = int(center + a)
        
        # Calculate weights for indices in range [start_idx, end_idx]
        valid_indices = []
        w_values = []
        
        for idx in range(start_idx, end_idx + 1):
            if 0 <= idx < C:
                dist = float_idx - idx
                # Manual sinc calc or use helper
                # Using Torch helper on scalar is bit overhead, use math/numpy here then tensor
                # Actually let's just do it in numpy for weight matrix construction
                if abs(dist) < a:
                    val = np.sinc(dist) * np.sinc(dist / a)
                    valid_indices.append(idx)
                    w_values.append(val)
        
        if len(valid_indices) > 0:
            w_values = np.array(w_values)
            w_sum = w_values.sum()
            if w_sum != 0:
                w_values /= w_sum
                
            # Fill matrix
            weight_matrix[valid_indices, j] = torch.from_numpy(w_values).to(device, dtype=torch.float32)

    # Perform Resampling (Matrix Mul)
    # (N, C_in) @ (C_in, C_out) -> (N, C_out)
    # print("  [GPU] Applying matrix multiplication...")
    resampled_flat = torch.matmul(flat_image, weight_matrix)
    
    return resampled_flat.reshape(H, W, C_out)

def save_patches(image, prefix, dataset_name=None, output_dir=None, valid_mask=None):
    # image can be numpy or tensor. If tensor, move to CPU for easy slicing/saving or keep GPU?
    # Keeping huge image on GPU might OOM if we accumulate patches.
    # Let's move to CPU first to be safe with memory, iterating on CPU is fast enough for slicing.
    if output_dir is None:
        output_dir = OUTPUT_DIR
    
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
        
    H, W, C = image.shape

    if valid_mask is not None:
        if isinstance(valid_mask, torch.Tensor):
            valid_mask = valid_mask.cpu().numpy()
        valid_mask = valid_mask.astype(bool)
        if valid_mask.shape != (H, W):
            print(
                f"  [WARNING] Invalid valid_mask shape {valid_mask.shape} for image {(H, W)}. Ignore mask.",
                flush=True
            )
            valid_mask = None
    
    # Save statistics for verification (Print ONLY ONCE per dataset)
    if dataset_name:
        with PRINT_LOCK:
            if dataset_name not in PRINTED_DATASETS:
                print(f"  [Patching] Input Stats ({dataset_name}) - Min: {image.min():.4f}, Max: {image.max():.4f}, Mean: {image.mean():.4f}", flush=True)
                PRINTED_DATASETS.add(dataset_name)
    # ...
    # print(f"  > Slicing grid: {len(h_starts)}x{len(w_starts)} = ~{total_expected} patches...", flush=True)

    # Use Non-overlapping logic (Stride=32)
    STRIDE = PATCH_SIZE
    
    h_starts = list(range(0, H - PATCH_SIZE + 1, STRIDE))
    if H % PATCH_SIZE != 0: h_starts.append(H - PATCH_SIZE)
        
    w_starts = list(range(0, W - PATCH_SIZE + 1, STRIDE))
    if W % PATCH_SIZE != 0: w_starts.append(W - PATCH_SIZE)
        
    h_starts = sorted(list(set(h_starts)))
    w_starts = sorted(list(set(w_starts)))
        
    count = 0
    total_expected = len(h_starts) * len(w_starts)
    # print(f"  > Slicing grid: {len(h_starts)}x{len(w_starts)} = ~{total_expected} patches...", flush=True)

    for h_start in h_starts:
        for w_start in w_starts:
            patch = image[h_start:h_start+PATCH_SIZE, w_start:w_start+PATCH_SIZE, :]

            if valid_mask is not None:
                patch_valid = valid_mask[h_start:h_start+PATCH_SIZE, w_start:w_start+PATCH_SIZE]
                if patch_valid.size == 0:
                    continue
                if float(patch_valid.mean()) < MIN_PATCH_VALID_RATIO:
                    continue
            
            # Check if patch is informative.
            # Some datasets contain negative offsets; mean-based filtering can wrongly drop valid patches.
            patch_max = float(patch.max())
            patch_min = float(patch.min())
            if patch_max > 1e-4 and (patch_max - patch_min) > 1e-4:
                # Clamp to [0, 1] to fix Lanczos resampling overshoot (e.g. 1.03)
                patch = np.clip(patch, 0.0, 1.0)
                patch_tensor = torch.from_numpy(patch.transpose(2, 0, 1)).half()
                
                # Check output dir exists
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                save_name = f"{prefix}_p{count:05d}.pt"
                save_path = os.path.join(output_dir, save_name)
                torch.save(patch_tensor, save_path)
                count += 1
            
    # print(f"鉁?Finished {prefix}: Saved {count} patches.", flush=True)
    return count

def sync_hyspecnet224ext_cache_to_temp(overwrite=False):
    """
    Copy cached HySpecNet224Ext patches to OUTPUT_DIR when cache and output differ.
    Returns copied patch count.
    """
    if os.path.abspath(HYSPECNET224EXT_WORKING_CACHE_DIR) == os.path.abspath(OUTPUT_DIR):
        return 0

    os.makedirs(HYSPECNET224EXT_WORKING_CACHE_DIR, exist_ok=True)
    cached_files = sorted(glob.glob(os.path.join(HYSPECNET224EXT_WORKING_CACHE_DIR, HYSPECNET224EXT_CACHE_PATTERN)))
    if not cached_files:
        return 0

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    copied = 0
    for src in tqdm(cached_files, desc="  Syncing HySpecNet224Ext cache", unit="pt"):
        dst = os.path.join(OUTPUT_DIR, os.path.basename(src))
        if (not overwrite) and os.path.exists(dst):
            continue
        try:
            shutil.copy2(src, dst)
            copied += 1
        except Exception as exc:
            print(f"  [WARNING] Failed to sync cache file {src}: {exc}", flush=True)
    return copied

# ================= Main Execution =================

def main():
    def bytes_to_gb(num_bytes):
        return float(num_bytes) / (1024 ** 3)
    
    def print_dataset_summary(dataset_name, scanned_images, sliced_patches, extra_note=None):
        msg = (
            f"  [Summary] {dataset_name}: scanned images={int(scanned_images)}, "
            f"sliced patches={int(sliced_patches)}"
        )
        if extra_note:
            msg += f" ({extra_note})"
        print(msg, flush=True)

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        print(f"Created output directory: {OUTPUT_DIR}")
        
    total_patches = 0
    total_raw_bytes = 0
        
    debug_priority_order = [
        'PotatoWaterStress',
        'HyperspectralBlueberries',
        'SpectroFood_FX10',
        'Cabbage_Eggplant',
        'WeedHSI_Reflectance',
        'SpectroFood_Apple',
        'SpectroFood_Broccoli',
        'SpectroFood_Mushroom'
    ]
    dataset_items = sorted(
        DATASET_CONFIG.items(),
        key=lambda kv: (
            0 if kv[0] in debug_priority_order else 1,
            debug_priority_order.index(kv[0]) if kv[0] in debug_priority_order else 999
        )
    )

    for dataset_name, config in dataset_items:
        print(f"\nProcessing {dataset_name}...", flush=True)
        path_config = config['path']
        dataset_scanned_images = 0
        dataset_sliced_patches = 0

        if not config.get('enabled', True):
            print(f"  [Skip] {dataset_name} is disabled by config (enabled=False).", flush=True)
            print_dataset_summary(
                dataset_name,
                dataset_scanned_images,
                dataset_sliced_patches,
                extra_note="disabled"
            )
            continue
        
        # Handle path as list or single string
        if isinstance(path_config, list):
            paths = path_config
        else:
            paths = [path_config]

        dataset_output_dir = OUTPUT_DIR
        if dataset_name == HYSPECNET224EXT_DATASET:
            os.makedirs(HYSPECNET224EXT_WORKING_CACHE_DIR, exist_ok=True)
            dataset_output_dir = HYSPECNET224EXT_WORKING_CACHE_DIR
            cache_in_output_root = os.path.abspath(dataset_output_dir) == os.path.abspath(OUTPUT_DIR)
            existing_cache = len(
                glob.glob(os.path.join(HYSPECNET224EXT_WORKING_CACHE_DIR, HYSPECNET224EXT_CACHE_PATTERN))
            )
            if existing_cache > 0:
                if cache_in_output_root:
                    print(
                        f"  [Cache] Existing cache hit in output root: {existing_cache}. "
                        f"Skip slicing {dataset_name}.",
                        flush=True
                    )
                    cache_note = "cache hit in output root"
                else:
                    synced_existing = sync_hyspecnet224ext_cache_to_temp(overwrite=False)
                    print(
                        f"  [Cache] Existing working cache hit: {existing_cache}. "
                        f"Synced {synced_existing} new files to {OUTPUT_DIR}. "
                        f"Skip slicing {dataset_name}.",
                        flush=True
                    )
                    cache_note = f"cache hit from working, synced={synced_existing}"
                total_patches += existing_cache
                dataset_sliced_patches = existing_cache
                print_dataset_summary(
                    dataset_name,
                    dataset_scanned_images,
                    dataset_sliced_patches,
                    extra_note=cache_note
                )
                continue

            if cache_in_output_root:
                print(
                    f"  [Cache] No reusable cache in output root. "
                    f"New {dataset_name} patches will be saved to: {dataset_output_dir}.",
                    flush=True
                )
            else:
                print(
                    f"  [Cache] No reusable working cache. "
                    f"New {dataset_name} patches will be saved to: {dataset_output_dir}. "
                    f"They will be synced to {OUTPUT_DIR} immediately after slicing.",
                    flush=True
                )
        
        # Check if any path is a directory (for file-based datasets)
        is_directory_dataset = any(os.path.isdir(p) for p in paths if os.path.exists(p))
        
        # Pre-fetch wavelengths to help with dimension checking
        source_wavs = get_source_wavelengths(dataset_name)
        expected_bands = len(source_wavs)

        # Define process_single_file here to capture `source_wavs` and `expected_bands`
        # This function will handle loading, normalizing, resampling, and saving for a single file.
        def process_single_file(fpath, key_name, config_item, ds_name, source_wavelengths_list):
            if os.path.basename(fpath) in BLACKLIST_FILES:
                return 0
            try:
                def parse_envi_header_numeric(hdr_path, field_name):
                    """Parse simple numeric scalar from ENVI .hdr, e.g. gain/ceiling."""
                    if not hdr_path or not os.path.exists(hdr_path):
                        return None
                    pat = re.compile(rf"\b{re.escape(field_name)}\b\s*=\s*([+-]?\d+(?:\.\d+)?)", re.IGNORECASE)
                    try:
                        with open(hdr_path, 'r', encoding='utf-8', errors='ignore') as fp:
                            for line in fp:
                                m = pat.search(line)
                                if m:
                                    return float(m.group(1))
                    except Exception:
                        return None
                    return None

                def normalize_preferred_key(preferred_key):
                    if preferred_key is None:
                        return None
                    if isinstance(preferred_key, str) and preferred_key.strip().lower() in ('none', ''):
                        return None
                    return preferred_key

                def pick_mat_cube_key_from_entries(entries, preferred_key, expected_band_count):
                    """
                    Pick the most plausible 3D cube key from metadata entries.
                    entries: iterable of (name, shape_tuple)
                    """
                    preferred_key = normalize_preferred_key(preferred_key)
                    if preferred_key is not None:
                        for k, shape in entries:
                            if k == preferred_key and len(shape) == 3:
                                return preferred_key

                    candidates = []
                    for k, shape in entries:
                        if len(shape) != 3:
                            continue

                        dim_diffs = [abs(int(dim) - int(expected_band_count)) for dim in shape]
                        min_diff = min(dim_diffs)
                        c_idx = dim_diffs.index(min_diff)
                        spatial_dims = [shape[i] for i in range(3) if i != c_idx]
                        spatial_area = int(spatial_dims[0]) * int(spatial_dims[1])
                        # Prefer exact band match first, then larger spatial area.
                        candidates.append((min_diff, -spatial_area, k))

                    if not candidates:
                        return None

                    candidates.sort()
                    return candidates[0][2]

                def pick_mat_cube_key_from_dict(mat_dict, preferred_key, expected_band_count):
                    """Pick the most plausible 3D numeric cube key from an in-memory .mat dictionary."""
                    preferred_key = normalize_preferred_key(preferred_key)
                    if preferred_key is not None and preferred_key in mat_dict:
                        arr = mat_dict[preferred_key]
                        if isinstance(arr, np.ndarray) and arr.ndim == 3 and np.issubdtype(arr.dtype, np.number):
                            return preferred_key

                    candidates = []
                    for k, v in mat_dict.items():
                        if k.startswith('__'):
                            continue
                        if not isinstance(v, np.ndarray):
                            continue
                        if v.ndim != 3:
                            continue
                        if not np.issubdtype(v.dtype, np.number):
                            continue

                        shape = v.shape
                        dim_diffs = [abs(int(dim) - int(expected_band_count)) for dim in shape]
                        min_diff = min(dim_diffs)
                        c_idx = dim_diffs.index(min_diff)
                        spatial_dims = [shape[i] for i in range(3) if i != c_idx]
                        spatial_area = int(spatial_dims[0]) * int(spatial_dims[1])
                        candidates.append((min_diff, -spatial_area, k))

                    if not candidates:
                        return None
                    candidates.sort()
                    return candidates[0][2]

                # Check for Samson header fix
                if ds_name == 'Samson' and config_item.get('needs_header_fix', False):
                    # This will create a .bin and .hdr from the .mat file
                    # and return the .hdr path. We then process the .hdr.
                    fpath = create_temp_samson_header(fpath)
                    # After creating the header, the original .mat file is no longer the target.
                    # The new fpath is the .hdr, which will be handled by the ENVI loader.

                img = None
                hdr_for_metadata = None
                # Load
                if fpath.lower().endswith('.mat'):
                    preferred_key = normalize_preferred_key(key_name)
                    selected_key = None
                    if config_item.get('mat_selective_load', False):
                        # Memory-safe path for very large MAT files (FX10):
                        # choose key via metadata, then load only that variable.
                        try:
                            mat_info = scipy.io.whosmat(fpath)
                            meta_entries = [(k, tuple(shape)) for k, shape, _ in mat_info]
                            selected_key = pick_mat_cube_key_from_entries(meta_entries, preferred_key, expected_bands)
                            if selected_key is None:
                                raise ValueError(f"No 3D numeric data key found in .mat file: {fpath}")
                            d = scipy.io.loadmat(fpath, variable_names=[selected_key])
                            if selected_key not in d:
                                raise ValueError(f"Key '{selected_key}' not found after selective load: {fpath}")
                            img = np.asarray(d[selected_key], dtype=np.float32)
                            del d
                        except NotImplementedError:
                            # MATLAB v7.3: use h5py fallback and only read selected dataset.
                            with h5py.File(fpath, 'r') as f:
                                h5_entries = []
                                for k in f.keys():
                                    obj = f[k]
                                    if isinstance(obj, h5py.Dataset):
                                        h5_entries.append((k, tuple(obj.shape)))
                                selected_key = pick_mat_cube_key_from_entries(h5_entries, preferred_key, expected_bands)
                                if selected_key is None:
                                    raise ValueError(f"No 3D numeric dataset key found in MAT v7.3 file: {fpath}")
                                img = np.array(f[selected_key], dtype=np.float32)
                    else:
                        # Default path for normal MAT files (faster, unchanged behavior).
                        try:
                            d = scipy.io.loadmat(fpath)
                            selected_key = pick_mat_cube_key_from_dict(d, preferred_key, expected_bands)
                            if selected_key is None:
                                raise ValueError(f"No 3D numeric data key found in .mat file: {fpath}")
                            img = np.asarray(d[selected_key], dtype=np.float32)
                            del d
                        except NotImplementedError:
                            with h5py.File(fpath, 'r') as f:
                                h5_entries = []
                                for k in f.keys():
                                    obj = f[k]
                                    if isinstance(obj, h5py.Dataset):
                                        h5_entries.append((k, tuple(obj.shape)))
                                selected_key = pick_mat_cube_key_from_entries(h5_entries, preferred_key, expected_bands)
                                if selected_key is None:
                                    raise ValueError(f"No 3D numeric dataset key found in MAT v7.3 file: {fpath}")
                                img = np.array(f[selected_key], dtype=np.float32)

                    # Align to HWC if channel dimension is first.
                    if img.ndim == 3 and img.shape[0] == expected_bands and img.shape[0] != img.shape[2]:
                        img = img.transpose(1, 2, 0)

                    if selected_key is None:
                        raise ValueError(f"No 3D numeric data key found in .mat file: {fpath}")
                    if preferred_key and selected_key != preferred_key:
                        print(
                            f"  [INFO] Preferred key '{preferred_key}' missing/invalid. "
                            f"Using '{selected_key}' for {os.path.basename(fpath)}.",
                            flush=True
                        )
                elif fpath.lower().endswith('.npy'):
                    img = np.load(fpath).astype(np.float32)
                elif fpath.lower().endswith(('.h5', '.h5py')):
                    # HDF5 Support (Neon)
                    with h5py.File(fpath, 'r') as f:
                        # Find data tensor
                        data = None
                        if ds_name == 'EarthView-Neon' and '1m' in f:
                            data = f['1m']
                        else:
                            def find_data(name, obj):
                                nonlocal data
                                if data is not None:
                                    return
                                if isinstance(obj, h5py.Dataset) and obj.ndim >= 3:
                                    data = obj
                            f.visititems(find_data)
                        
                        if data is None: return 0
                        img = np.array(data).astype(np.float32)
                        
                        # Handle 4D cubes.
                        if img.ndim == 4:
                            if ds_name == 'EarthView-Neon':
                                # EarthView-Neon '/1m' is (T, C, H, W), e.g. (3, 369, 64, 64).
                                if img.shape[1] > img.shape[2] and img.shape[1] > img.shape[3]:
                                    t, c, h, w = img.shape
                                    img = np.transpose(img, (0, 2, 3, 1)).reshape(t * h, w, c)
                                # Fallback for possible (T, H, W, C)
                                elif img.shape[3] > img.shape[1] and img.shape[3] > img.shape[2]:
                                    t, h, w, c = img.shape
                                    img = img.reshape(t * h, w, c)
                                else:
                                    raise ValueError(f"Unexpected EarthView-Neon H5 shape: {img.shape}")
                            else:
                                # Generic fallback for other 4D data.
                                imgs = [img[i] for i in range(img.shape[0])]
                                img = np.concatenate(imgs, axis=1)  # (C, N*H, W)
                elif fpath.lower().endswith(('.tif', '.tiff')):
                    img = tifffile.imread(fpath).astype(np.float32)
                elif fpath.lower().endswith(('.img', '.hdr', '.dat', '.bil')):
                    try:
                        import spectral.io.envi as envi
                    except ImportError:
                        print("  > 'spectral' library not found. Installing...", flush=True)
                        subprocess.check_call([sys.executable, "-m", "pip", "install", "spectral"])
                        import spectral.io.envi as envi
                    
                    base = os.path.splitext(fpath)[0]
                    hdr_file = None
                    img_file = None

                    # Check for explicit header in config
                    if config_item.get('header_path'):
                        hdr_file = config_item['header_path']
                        # Assume current fpath is the image if it's not the header itself
                        if not fpath.lower().endswith('.hdr'):
                             img_file = fpath
                        else:
                             # accessing via header logic?
                             # In directory mode we might hit .hdr first.
                             # But Samson is single file mode.
                             img_file = config_item['path'] # or fpath
                    
                    if not hdr_file:
                         if fpath.lower().endswith('.hdr'):
                            hdr_file = fpath
                            # Try to find associated binary
                            candidates = [base + '.img', base + '.dat', base + '.bil', base + '.bip', base + '.bin', base]
                            for c in candidates:
                                if os.path.exists(c):
                                    img_file = c
                                    break
                            if not img_file:
                                print(f"  Error: Binary file for header {fpath} not found. Skipping.", flush=True)
                                return 0
                         elif fpath.lower().endswith(('.img', '.dat', '.bil')):
                            img_file = fpath
                            # Try to find associated header
                            candidates = [fpath + '.hdr', base + '.hdr', base + '.top'] # supports *.bil.hdr
                            for c in candidates:
                                if os.path.exists(c):
                                    hdr_file = c
                                    break
                            if not hdr_file:
                                print(f"  Error: Header file for image {fpath} not found. Skipping.", flush=True)
                                return 0
                    
                    if hdr_file and img_file:
                        if not os.path.exists(hdr_file):
                             print(f"  Error: Explicit header file not found at {hdr_file}", flush=True)
                             return 0
                        
                        hdr_for_metadata = hdr_file
                        print(f"  > Loading ENVI: {img_file} with header {hdr_file}", flush=True)
                        img_obj = envi.open(hdr_file, img_file)
                        img = img_obj.load().astype(np.float32)
                    else:
                        print(f"  Error: Could not determine ENVI files for {fpath}. Skipping.", flush=True)
                        return 0
                else:
                    raise ValueError(f"Unsupported file type: {fpath}")
                        
                # Ensure (H, W, C) format
                if img.ndim == 3:
                    if img.shape[0] == expected_bands and img.shape[0] < img.shape[2]: # (C, H, W)
                        img = img.transpose(1, 2, 0)
                    elif img.shape[2] != expected_bands and img.shape[0] == expected_bands: # (C, H, W) but C is not last
                        img = img.transpose(1, 2, 0) # Assume C is first if it matches expected_bands
                    # Else assume (H, W, C) or (H, C, W) which will be caught by shape check later

                if img.ndim != 3:
                    print(f"  Error: Unsupported image ndim={img.ndim} for {fpath}. Skipping.", flush=True)
                    return 0

                # Dataset-specific per-file calibration from HDR metadata.
                if ds_name == 'HyperspectralBlueberries' and hdr_for_metadata:
                    if config_item.get('apply_gain_correction', False):
                        gain_val = parse_envi_header_numeric(hdr_for_metadata, 'gain')
                        ref_gain = float(config_item.get('gain_reference', 5.0))
                        if gain_val is not None and gain_val > 0 and ref_gain > 0:
                            gain_scale = ref_gain / gain_val
                            if abs(gain_scale - 1.0) > 1e-6:
                                img = (img * gain_scale).astype(np.float32, copy=False)
                                print(
                                    f"  [INFO] Blueberries gain correction: "
                                    f"gain={gain_val:.2f} -> x{gain_scale:.4f} for {os.path.basename(fpath)}",
                                    flush=True
                                )
                    if config_item.get('use_hdr_ceiling', False):
                        ceiling_val = parse_envi_header_numeric(hdr_for_metadata, 'ceiling')
                        if ceiling_val is not None and ceiling_val > 0:
                            img = (img / ceiling_val).astype(np.float32, copy=False)

                # For strictly defined sensors, reject cubes with unexpected channel count.
                if ds_name in STRICT_BAND_DATASETS and img.shape[2] != expected_bands:
                    print(
                        f"  [WARNING] Skip {os.path.basename(fpath)} for {ds_name}: "
                        f"expected {expected_bands} bands, got {img.shape[2]}.",
                        flush=True
                    )
                    return 0

                source_wavs_for_img = ensure_wavelength_count(source_wavelengths_list, img.shape[2], ds_name)

                valid_mask_for_patches = None
                invalid_voxel = None
                if ds_name == HYSPECNET224EXT_DATASET:
                    # EnMAP L2A nodata (commonly -9999) should not participate in filtering/normalization.
                    invalid_voxel = ~np.isfinite(img)
                    invalid_voxel |= np.abs(img - NODATA_VALUE) <= NODATA_TOL
                    invalid_voxel |= img < (NODATA_VALUE + NODATA_TOL)

                    # Keep wavelength alignment strict: remove only fixed HySpecNet water-vapor bands.
                    # This follows hyspecnet11k note indices (0-based): [127-141], [161-167].
                    if img.shape[2] == HYSPECNET224_BAND_COUNT:
                        keep_band_mask = np.ones(img.shape[2], dtype=bool)
                        keep_band_mask[HYSPECNET_WATER_VAPOR_REMOVE_INDICES] = False
                        img = img[:, :, keep_band_mask]
                        source_wavs_for_img = source_wavs_for_img[keep_band_mask]
                        invalid_voxel = invalid_voxel[:, :, keep_band_mask]

                    pixel_valid_ratio = 1.0 - invalid_voxel.mean(axis=2)
                    valid_mask_for_patches = pixel_valid_ratio >= MIN_PATCH_VALID_RATIO
                    if float(valid_mask_for_patches.mean()) < 1e-4:
                        return 0
                else:
                    nodata_value = config_item.get('nodata_value', None)
                    nodata_tol = float(config_item.get('nodata_tol', 0.0))
                    # Potato-only fallback: read ENVI "data ignore value" directly from header.
                    if ds_name == 'PotatoWaterStress' and nodata_value is None and hdr_for_metadata:
                        nodata_value = parse_envi_header_numeric(hdr_for_metadata, 'data ignore value')
                    if nodata_value is not None:
                        nodata_value = float(nodata_value)
                        invalid_voxel = ~np.isfinite(img)
                        if nodata_tol > 0:
                            invalid_voxel |= np.abs(img - nodata_value) <= nodata_tol
                        else:
                            invalid_voxel |= (img == nodata_value)

                        pixel_valid_ratio = 1.0 - invalid_voxel.mean(axis=2)
                        valid_mask_for_patches = pixel_valid_ratio >= MIN_PATCH_VALID_RATIO
                        if float(valid_mask_for_patches.mean()) < 1e-4:
                            return 0
                    elif not np.isfinite(img).all():
                        img = np.nan_to_num(img, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

                # Optional dataset-specific pre-clip (raw domain), e.g. remove hot/dead pixels.
                pre_clip = config_item.get('pre_clip_range')
                if pre_clip is not None:
                    try:
                        clip_low = float(pre_clip[0])
                        clip_high = float(pre_clip[1])
                        if clip_high > clip_low:
                            img = np.clip(img, clip_low, clip_high).astype(np.float32, copy=False)
                    except Exception:
                        print(f"  [WARNING] Invalid pre_clip_range={pre_clip} for {ds_name}. Ignored.", flush=True)
                
                # Guardrail: reflectance datasets should not contain extreme numeric ranges.
                if ds_name in REFLECTANCE_SANITY_DATASETS:
                    # "Too many non-finite" should only reflect NaN/Inf, not configured nodata values.
                    finite_only_mask = np.isfinite(img)
                    if float(finite_only_mask.mean()) < 0.99:
                        print(
                            f"  [WARNING] Too many non-finite values in {os.path.basename(fpath)} ({ds_name}). Skipping.",
                            flush=True
                        )
                        return 0
                    if invalid_voxel is not None:
                        valid_vals = img[(~invalid_voxel) & finite_only_mask]
                    else:
                        valid_vals = img[finite_only_mask]
                    if valid_vals.size == 0:
                        print(
                            f"  [WARNING] No valid values after nodata masking in {os.path.basename(fpath)} ({ds_name}). Skipping.",
                            flush=True
                        )
                        return 0
                    p01 = float(np.percentile(valid_vals, 0.1))
                    p99 = float(np.percentile(valid_vals, 99.9))
                    if p01 < -1.0 or p99 > 10.0:
                        print(
                            f"  [WARNING] Abnormal value range in {os.path.basename(fpath)} ({ds_name}): "
                            f"p0.1={p01:.4f}, p99.9={p99:.4f}. Skipping.",
                            flush=True
                        )
                        return 0
                
                # Avoid mean-based rejection: some valid cubes have negative global mean.
                if invalid_voxel is not None:
                    valid_values = img[~invalid_voxel]
                    if valid_values.size == 0:
                        return 0
                    img_max = float(valid_values.max())
                    img_min = float(valid_values.min())
                else:
                    img_max = float(img.max())
                    img_min = float(img.min())
                img_range = img_max - img_min
                if img_max < 1e-5 and img_min > -1e-5:
                    return 0
                if img_range < 1e-5:
                    return 0
                        
                # Normalize
                if config_item.get('normalize', False):
                    if invalid_voxel is not None:
                        valid_values = img[~invalid_voxel]
                        p_low, p_high = np.percentile(valid_values, [2, 98])
                        img = img.astype(np.float32, copy=True)
                        if p_high > p_low:
                            clipped = np.clip(valid_values, p_low, p_high)
                            img[~invalid_voxel] = ((clipped - p_low) / (p_high - p_low)).astype(np.float32)
                    else:
                        p_low, p_high = np.percentile(img, [2, 98])
                        img = np.clip(img, p_low, p_high).astype(np.float32)
                        if p_high > p_low:
                            img = ((img - p_low) / (p_high - p_low)).astype(np.float32)
                if config_item['scale_factor'] != 1.0:
                    img = img.astype(np.float32, copy=False)
                    if invalid_voxel is not None:
                        img[~invalid_voxel] /= config_item['scale_factor']
                    else:
                        img /= config_item['scale_factor']

                # Fill nodata after normalization/scaling to avoid contaminating statistics.
                if invalid_voxel is not None:
                    img = np.where(invalid_voxel, 0.0, img).astype(np.float32)
                else:
                    img = img.astype(np.float32, copy=False)
                        
                # Resample
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                img_t = torch.from_numpy(img).to(device)
                res = lanczos_resample_gpu(img_t, source_wavs_for_img, TARGET_WAVELENGTHS, a=2)
                    
                # Save
                import hashlib
                path_hash = hashlib.md5(fpath.encode()).hexdigest()[:6]
                save_name = f"{ds_name}_{os.path.basename(fpath).split('.')[0]}_{path_hash}"
                patch_count = save_patches(
                    res,
                    save_name,
                    dataset_name=ds_name,
                    output_dir=dataset_output_dir,
                    valid_mask=valid_mask_for_patches
                )
                # Release large buffers promptly before next file.
                del res, img_t, img
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return patch_count
            except Exception as e:
                print(f"Err processing {fpath}: {e}", flush=True)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return 0

        if is_directory_dataset:
            # Directory Processing - collect files from all paths
            all_files_to_process = []
            for path in paths:
                if not os.path.exists(path):
                    print(f"  Warning: Path not found: {path}. Skipping this path.", flush=True)
                    continue
                
                if os.path.isdir(path):
                    print(f"  > Scanning directory: {path}...", flush=True)
                    # Use recursive glob for robust finding (handle subfolders)
                    all_files = glob.glob(os.path.join(path, "**"), recursive=True)
                    # Added .dat and .img for ENVI support
                    allowed_exts = ('.tif', '.tiff', '.mat', '.npy', '.h5', '.h5py', '.img', '.dat', '.bil', '.hdr')
                    dir_files = [f for f in all_files if f.lower().endswith(allowed_exts) and os.path.isfile(f)]
                    
                    # Apply config-based filters
                    tgt_kw = config.get('filter_kw', None)
                    ign_kws = config.get('ignore_kws', [])
                    
                    if tgt_kw:
                        dir_files = [f for f in dir_files if tgt_kw in f]
                    
                    for k in ign_kws:
                        dir_files = [f for f in dir_files if k not in f]

                    if len(dir_files) == 0:
                        print(f"    No files found in {path} matching extensions {allowed_exts}", flush=True)
                        continue
                        
                    print(f"    Found {len(dir_files)} files in {path} (filtered)", flush=True)
                    all_files_to_process.extend(dir_files)
            
            if len(all_files_to_process) == 0:
                print(f"[DEBUG] No files found in any of the paths for {dataset_name}")
                print_dataset_summary(
                    dataset_name,
                    dataset_scanned_images,
                    dataset_sliced_patches,
                    extra_note="no eligible files"
                )
                continue

            dataset_scanned_images = len(all_files_to_process)
            dataset_raw_bytes = 0
            for f in all_files_to_process:
                try:
                    dataset_raw_bytes += os.path.getsize(f)
                except OSError:
                    pass
            total_raw_bytes += dataset_raw_bytes
            print(
                f"  [DataVolume] Raw scanned size ({dataset_name}): {bytes_to_gb(dataset_raw_bytes):.2f} GB",
                flush=True
            )
            
            # Use configurable thread workers (safe with CUDA).
            from concurrent.futures import ThreadPoolExecutor

            # Define process_single_file here to capture `source_wavs` and `expected_bands`
            def process_single_file_wrapper(fpath):
                # Pass necessary config context
                return process_single_file(fpath, config.get('key'), config, dataset_name, source_wavs)

            dataset_workers = config.get('num_workers', NUM_WORKERS)
            try:
                dataset_workers = max(1, int(dataset_workers))
            except Exception:
                dataset_workers = NUM_WORKERS

            # Hard guarantee: process large/special datasets strictly one file at a time.
            if dataset_name in {'HyperspectralBlueberries', 'PotatoWaterStress'}:
                dataset_workers = 1

            print(f"  > Starting processing with {dataset_workers} Threads...", flush=True)
            if dataset_name in {'HyperspectralBlueberries', 'PotatoWaterStress'} and dataset_workers == 1:
                results = []
                for fpath in tqdm(all_files_to_process, total=len(all_files_to_process), desc=f"  Processing {dataset_name} (Serial)", unit="img"):
                    results.append(process_single_file_wrapper(fpath))
            else:
                with ThreadPoolExecutor(max_workers=dataset_workers) as executor:
                   results = list(tqdm(executor.map(process_single_file_wrapper, all_files_to_process), total=len(all_files_to_process), desc=f"  Processing {dataset_name} (Parallel)", unit="img"))
            
            dataset_sliced_patches = int(sum(results))
            total_patches += dataset_sliced_patches
            if dataset_name == HYSPECNET224EXT_DATASET:
                synced_now = sync_hyspecnet224ext_cache_to_temp(overwrite=True)
                working_total = len(glob.glob(os.path.join(HYSPECNET224EXT_WORKING_CACHE_DIR, HYSPECNET224EXT_CACHE_PATTERN)))
                print(
                    f"  [Cache] Post-slice sync done. Working cache: {working_total} files. "
                    f"Synced to temp this pass: {synced_now}",
                    flush=True
                )
            print_dataset_summary(dataset_name, dataset_scanned_images, dataset_sliced_patches)
            continue # Done with directory

        # Standard single-file handling (non-directory datasets)
        existing_file_paths = [p for p in paths if os.path.exists(p) and os.path.isfile(p)]
        if not existing_file_paths:
            print(f"Warning: File not found at any candidate path: {paths}. Skipping.", flush=True)
            print_dataset_summary(
                dataset_name,
                dataset_scanned_images,
                dataset_sliced_patches,
                extra_note="source file missing"
            )
            continue
        mat_path = existing_file_paths[0]

        dataset_scanned_images = 1
        try:
            single_raw_bytes = os.path.getsize(mat_path)
        except OSError:
            single_raw_bytes = 0
        total_raw_bytes += single_raw_bytes
        print(
            f"  [DataVolume] Raw scanned size ({dataset_name}): {bytes_to_gb(single_raw_bytes):.2f} GB",
            flush=True
        )
            
        try:
            print(f"  > Loading file from {mat_path}...", flush=True)
            image_raw = None
            
            # --- 0. ENVI binary without extension (e.g. CE_Reflectance_Data + .hdr) ---
            if (
                os.path.isfile(mat_path)
                and os.path.splitext(mat_path)[1] == ''
                and os.path.exists(mat_path + '.hdr')
            ):
                try:
                    import spectral.io.envi as envi
                except ImportError:
                    print("  > 'spectral' library not found. Installing...", flush=True)
                    subprocess.check_call([sys.executable, "-m", "pip", "install", "spectral"])
                    import spectral.io.envi as envi
                hdr_file = mat_path + '.hdr'
                print(f"  > Found paired header for extensionless ENVI: {hdr_file}", flush=True)
                img_obj = envi.open(hdr_file, mat_path)
                image_raw = np.array(img_obj.load()).astype(np.float32)

            # --- 1. TIF/TIFF ---
            if image_raw is None and mat_path.lower().endswith(('.tif', '.tiff')):
                image_raw = tifffile.imread(mat_path).astype(np.float32)
                # Tiff usually (H, W, C) or (C, H, W). 
                # WashingtonDC is (1280, 307, 210) from description, let's check dim
                # If (C, H, W) -> Transpose to (H, W, C) for consistency
                if image_raw.ndim == 3 and image_raw.shape[0] < image_raw.shape[2]:
                     # Heuristic: Channels usually smaller? No, HS can have 200 channels.
                     # But WashingtonDC is 1280x307x210.
                     # tifffile often reads as is. 
                     # We will check expected bands later and transpose if needed.
                     pass

            # --- 2. IMG/HDR or DAT/HDR (ENVI) ---
            elif image_raw is None and mat_path.lower().endswith(('.img', '.hdr', '.dat', '.bil')):
                try:
                    import spectral.io.envi as envi
                except ImportError:
                    print("  > 'spectral' library not found. Installing...", flush=True)
                    subprocess.check_call([sys.executable, "-m", "pip", "install", "spectral"])
                    import spectral.io.envi as envi
                
                envi_path = mat_path
                if config.get('needs_header_fix', False):
                    try:
                        envi_path = create_temp_samson_header(mat_path)
                    except Exception as exc:
                        print(f"  Error: Failed to create temporary header for {mat_path}: {exc}", flush=True)

                if envi_path.lower().endswith('.hdr'):
                    # User passed the header file directly
                    base_no_ext = os.path.splitext(envi_path)[0]
                    candidates = [base_no_ext + '.img', base_no_ext + '.dat', base_no_ext + '.bil', base_no_ext + '.bip', base_no_ext + '.bin', base_no_ext]
                    img_bin = None
                    for c in candidates:
                        if os.path.exists(c):
                            img_bin = c
                            break

                    if img_bin:
                        try:
                            print(f"  > Found associated binary: {img_bin}", flush=True)
                            img_obj = envi.open(envi_path, img_bin)
                            image_raw = np.array(img_obj.load()).astype(np.float32)
                        except Exception as exc:
                            print(f"  Error loading binary {img_bin}: {exc}", flush=True)
                    else:
                        print(f"  Error: Binary file (.img/.dat/.bin/None) for header {envi_path} not found.", flush=True)
                else:
                    hdr_file = None
                    if config.get('header_path'):
                        if os.path.exists(config['header_path']):
                            hdr_file = config['header_path']
                        else:
                            print(f"  Warning: Explicit header not found: {config['header_path']}", flush=True)

                    if hdr_file is None:
                        base = os.path.splitext(envi_path)[0]
                        top_candidate = base + '.top'
                        hdr_candidate = base + '.hdr'
                        hdr_sidecar_candidate = envi_path + '.hdr'
                        if os.path.exists(hdr_sidecar_candidate):
                            hdr_file = hdr_sidecar_candidate
                        elif os.path.exists(top_candidate):
                            hdr_file = top_candidate
                        elif os.path.exists(hdr_candidate):
                            hdr_file = hdr_candidate

                    if hdr_file:
                        print(f"  > Found header: {hdr_file}", flush=True)
                        img_obj = envi.open(hdr_file, envi_path)
                        image_raw = np.array(img_obj.load()).astype(np.float32)
                    else:
                        print(f"  Error: Header .hdr not found for {envi_path}", flush=True)

            # --- 3. H5/H5PY ---
            elif image_raw is None and mat_path.lower().endswith(('.h5', '.h5py')):
                # HDF5 Support
                with h5py.File(mat_path, 'r') as f:
                    # Find data tensor
                    data = None
                    if dataset_name == 'EarthView-Neon' and '1m' in f:
                        data = f['1m']
                    else:
                        def find_data_h5(name, obj):
                            nonlocal data
                            if data is not None:
                                return
                            if isinstance(obj, h5py.Dataset) and obj.ndim >= 3:
                                data = obj
                        f.visititems(find_data_h5)
                    
                    if data is not None:
                        image_raw = np.array(data).astype(np.float32)
                        # Handle 4D cubes.
                        if image_raw.ndim == 4:
                            if dataset_name == 'EarthView-Neon':
                                # EarthView-Neon '/1m' is (T, C, H, W), e.g. (3, 369, 64, 64).
                                if image_raw.shape[1] > image_raw.shape[2] and image_raw.shape[1] > image_raw.shape[3]:
                                    t, c, h, w = image_raw.shape
                                    image_raw = np.transpose(image_raw, (0, 2, 3, 1)).reshape(t * h, w, c)
                                # Fallback for possible (T, H, W, C)
                                elif image_raw.shape[3] > image_raw.shape[1] and image_raw.shape[3] > image_raw.shape[2]:
                                    t, h, w, c = image_raw.shape
                                    image_raw = image_raw.reshape(t * h, w, c)
                                else:
                                    raise ValueError(f"Unexpected EarthView-Neon H5 shape: {image_raw.shape}")
                            else:
                                imgs = [image_raw[i] for i in range(image_raw.shape[0])]
                                image_raw = np.concatenate(imgs, axis=1) # (C, N*H, W) -> Transpose later?
            
            # --- 4. NPY ---
            elif image_raw is None and mat_path.lower().endswith('.npy'):
                 image_raw = np.load(mat_path).astype(np.float32)

            # --- 5. MAT (Fallback/Default) ---
            elif image_raw is None:
                 # Assume .mat
                 # Try scipy first, fallback to h5py for MATLAB v7.3 files
                try:
                    data = scipy.io.loadmat(mat_path)
                    key_name = config['key']
                    
                    # Dynamic key finding if specified key fails
                    if key_name not in data:
                         keys = [k for k in data.keys() if not k.startswith('__')]
                         key_name = keys[0]
                         print(f"  Key '{config['key']}' not found. Using '{key_name}' instead.", flush=True)
    
                    image_raw = data[key_name].astype(np.float32)
                except NotImplementedError:
                    # MATLAB v7.3 format - use h5py
                    print(f"  > Detected MATLAB v7.3 format, using h5py...", flush=True)
                    with h5py.File(mat_path, 'r') as f:
                        key_name = config['key']
                        if key_name not in f:
                            keys = [k for k in f.keys()]
                            key_name = keys[0]
                            print(f"  Key '{config['key']}' not found. Using '{key_name}' instead.", flush=True)
                        
                        # h5py returns data in transposed order (C, H, W) for MATLAB's (H, W, C)
                        image_raw = np.array(f[key_name]).astype(np.float32)
                        # Transpose from (C, W, H) to (H, W, C) - MATLAB stores column-major
                        if image_raw.ndim == 3:
                            image_raw = np.transpose(image_raw, (2, 1, 0))

            if image_raw is None:
                print(f"  Error: Failed to load data from {mat_path}", flush=True)
                continue

            # --- Post-Load Shape Check & Transpose ---
            # Ensure (H, W, C)
            # Heuristic: Band dimension matches expected_bands
            source_wavs = get_source_wavelengths(dataset_name)
            expected_bands = len(source_wavs)
            
            if image_raw.ndim == 3:
                if image_raw.shape[0] == expected_bands:
                     print(f"  > Transposing (C, H, W) {image_raw.shape} to (H, W, C)...", flush=True)
                     image_raw = np.transpose(image_raw, (1, 2, 0))
                elif image_raw.shape[2] == expected_bands:
                     # Already (H, W, C)
                     pass
                else:
                     print(f"  [WARNING] Dimension mismatch! Expected {expected_bands} bands. Got shape {image_raw.shape}. Guessing channel dim...", flush=True)
                     # Try to align
                     if image_raw.shape[0] == expected_bands: image_raw = np.transpose(image_raw, (1, 2, 0))
            else:
                print(f"  Error: Unsupported image ndim={image_raw.ndim} for {dataset_name}. Skipping.", flush=True)
                continue

            source_wavs = ensure_wavelength_count(source_wavs, image_raw.shape[2], dataset_name)

            pre_clip = config.get('pre_clip_range')
            if pre_clip is not None:
                try:
                    clip_low = float(pre_clip[0])
                    clip_high = float(pre_clip[1])
                    if clip_high > clip_low:
                        image_raw = np.clip(image_raw, clip_low, clip_high).astype(np.float32, copy=False)
                except Exception:
                    print(f"  [WARNING] Invalid pre_clip_range={pre_clip} for {dataset_name}. Ignored.", flush=True)
            
            # Print Statistics
            print(f"  Original shape: {image_raw.shape}", flush=True)
            stats_msg = f"  Raw stats: Min={image_raw.min():.4f}, Max={image_raw.max():.4f}, Mean={image_raw.mean():.4f}"
            if image_raw.size > 0:
                p2, p98 = np.percentile(image_raw, [2, 98])
                stats_msg += f", P2={p2:.4f}, P98={p98:.4f}"
            print(stats_msg, flush=True)
            
            # 1. Normalize (Robust)
            if config.get('normalize', False):
                 p_low, p_high = np.percentile(image_raw, [2, 98])
                 print(f"  Doing Robust Normalization (Clip 2%-98%): [{p_low:.4f}, {p_high:.4f}]...")
                 
                 # Clip first
                 image_raw = np.clip(image_raw, p_low, p_high).astype(np.float32)
                 
                 # Then Min-Max to 0-1
                 image_min = p_low
                 image_max = p_high
                 if image_max > image_min:
                     image_raw = ((image_raw - image_min) / (image_max - image_min)).astype(np.float32)
                 
                 # DEBUG: Verify dtype and range immediately after normalization
                 print(f"  [DEBUG] Post-Norm Stats: Mean={image_raw.mean():.4f}, Max={image_raw.max():.4f}, Dtype={image_raw.dtype}", flush=True)

            if config['scale_factor'] != 1.0:
                print(f"  Normalizing by factor {config['scale_factor']}...")
                image_raw /= config['scale_factor']
            
            # 2. Resample (GPU Accelerated)
            # Move to GPU
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            if device.type == 'cuda':
                print(f"  Using GPU ({torch.cuda.get_device_name(0)}) for resampling...", flush=True)
            
            image_tensor = torch.from_numpy(image_raw).to(device)
            
            # Resample on GPU
            image_resampled_tensor = lanczos_resample_gpu(image_tensor, source_wavs, TARGET_WAVELENGTHS, a=2)
            
            # Move back to CPU numpy for patching (or keep one patch on gpu? patching on cpu is easier for slicing logic if complex)
            # Actually our save_patches moves to torch.from_numpy, so we can pass tensor directly!
            # Let's adapt save_patches to accept tensor.
            
            # 3. Patching
            print(f"  Generating patches for {dataset_name}...", flush=True)
            import hashlib
            path_hash = hashlib.md5(mat_path.encode()).hexdigest()[:6]
            save_name_single = f"{dataset_name}_{os.path.basename(mat_path).split('.')[0]}_{path_hash}"
            dataset_sliced_patches = save_patches(
                image_resampled_tensor,
                save_name_single,
                dataset_name=dataset_name,
                output_dir=dataset_output_dir
            )
            total_patches += dataset_sliced_patches
            
        except Exception as e:
            print(f"Error processing {dataset_name}: {e}")
        print_dataset_summary(dataset_name, dataset_scanned_images, dataset_sliced_patches)

    print(f"\n==================================================", flush=True)
    print(f"All Done! Total patches generated: {total_patches}", flush=True)
    print(f"Total raw scanned data size: {bytes_to_gb(total_raw_bytes):.2f} GB", flush=True)
    print(f"==================================================", flush=True)

if __name__ == "__main__":
    main()



from scipy.integrate import IntegrationWarning
import warnings
#warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=IntegrationWarning)
#warnings.filterwarnings("ignore", message='.*cgi.*', category=DeprecationWarning)
#warnings.filterwarnings("ignore", message=".*'cgi' is deprecated.*", category=DeprecationWarning)

import os
os.environ["PYTHONWARNINGS"] = "ignore::DeprecationWarning"

import pandas as pd
#with warnings.catch_warnings():
#    warnings.filterwarnings("ignore", message=".*'cgi' is deprecated.*", category=DeprecationWarning)
#    import mne
import copy
import time
import json
import traceback
import math
import numpy as np
import sys
import multiprocessing
from multiprocessing import Process, Queue, Pool
import matplotlib.pyplot as plt
#%matplotlib inline
import nibabel as nib # common way of importing nibabel
import scipy.optimize as scopt
import numpy as np
import scipy
from scipy.optimize import LinearConstraint
from scipy import interpolate
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.linalg import svd, pinv
from scipy.optimize import SR1
from scipy import integrate
import concurrent.futures
from typing import List, Dict, Tuple
import pickle
from diploma_functions import get_X, load_patient_data, chunk_voxels, run_computing_on_chunk,\
computing, get_best_thresholds, get_reproducibility, get_map_array, plot_map
from diploma_functions import is_serializable, total_size, coord_voxel, f_obj, get_thresholds_dict, get_t_groups, \
voxel_coord, coord_voxel, load_patient_logs, print_matrix, print_vector, get_Y, NpEncoder

path_to_folder = r'C:\Users\andrew\Desktop\Mag Diploma\Код и новые данные\Запуск_3'
list_dir = os.listdir(path=path_to_folder)
params_files = {}
arrays = {}
contrast_names = []
for dir in list_dir:
    if dir.split('.')[1] == 'json':
        contrast_name = dir.split('.')[0].split('_')[1]
        if contrast_name not in contrast_names:
            contrast_names.append(contrast_name)
        params_files.append(dir)
    elif dir.split('.')[1] == 'npy':
        contrast_name = dir.split('.')[0].split('_')[1]
        if contrast_name not in contrast_names:
            contrast_names.append(contrast_name)
        arrays[contrast_name] = dir
for contrast_name in contrast_names:
    par_name = params_files[contrast_name]
    arr_name = arrays[contrast_name]
    path_to_params = os.path.join(path_to_folder, par_name)
    path_to_array = os.path.join(path_to_folder, arr_name)
    reproducibility_array = np.load(path_to_array)
    parameters_dict = json.load(path_to_params, cls=NpEncoder)
    print('contrast =', contrast_name)
    _ = plot_map(None, None, array_to_show=reproducibility_array)
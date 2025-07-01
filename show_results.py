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
computing, get_best_thresholds, get_reproducibility, get_map_array, plot_maps
from diploma_functions import is_serializable, total_size, coord_voxel, f_obj, get_thresholds_dict, get_t_groups, \
voxel_coord, coord_voxel, load_patient_logs, print_matrix, print_vector, get_Y, NpEncoder

path_to_folder = r'C:\Users\andrew\Desktop\Mag Diploma\Code_and_new_data\Запуск_92'
list_dir = os.listdir(path=path_to_folder)
params_files = {}
arrays = {}
contrast_names = []
arrays = {}
array_cont = []
array_ways = []
array_K = []
jsons = {}
json_cont = []
json_names = []
json_files = []
json_ways = []
json_K = []
json_levels = 1
array_levels = 1
array_names = []
array_files = []
#print(list_dir)
for dir in list_dir:
    if dir.split('.')[1] == 'json':
        json_names.append(dir)
        file = open(os.path.join(path_to_folder, dir))
        json_files.append(json.load(file))
    elif dir.split('.')[1] == 'npy':
        array_names.append(dir)
        array_files.append(np.load(os.path.join(path_to_folder, dir)))
'''
print(contrast_names)
print(arrays)
print(params_files)
if len(array_cont) == 0:
    array_cont.append(0)
if len(array_ways) == 0:
    array_ways.append(0)
if len(array_K) == 0:
    array_K.append(0)


if (array_levels, json_levels) == (1, 1):
    par_name = jsons['a']
    arr_name = arrays['a']
    path_to_params = os.path.join(path_to_folder, par_name)
    path_to_array = os.path.join(path_to_folder, arr_name)
    reproducibility_array = np.load(path_to_array)
    json_file = path_to_params
    file = open(json_file)
    parameters_dict = json.load(file, cls=NpEncoder)
    print('contrast =', contrast_name)
    _ = plot_map(None, None, array_to_show=reproducibility_array, save=False)


if (array_levels, json_levels) == (1, 2):


if (array_levels, json_levels) == (2, 1):


if (array_levels, json_levels) == (2, 2):

'''
'''


for contrast_name in array_cont:
    for way in array_ways:
        for K in array_K:
            par_name = jsons[contrast_name]
            arr_name = arrays[contrast_name]
            path_to_params = os.path.join(path_to_folder, par_name)
            path_to_array = os.path.join(path_to_folder, arr_name)
            reproducibility_array = np.load(path_to_array)
            json_file = path_to_params
            file = open(json_file)
            parameters_dict = json.load(file, cls=NpEncoder)
            print('contrast =', contrast_name)
            print('way =', way)
            print('K =', K)
            _ = plot_map(None, None, array_to_show=reproducibility_array, save=False)

'''
parameters = json_files[0]
patient = os.path.split(parameters['data_path'])[1]
anat_file_name = 'wm' + patient + '_T1.nii'
anat_path = os.path.join(parameters['data_path'], 'structural', anat_file_name)
anat_scan = nib.load(anat_path)
anat_map_array = anat_scan.get_fdata()

#print(array_files)
for i, file in enumerate(array_files):
    split = array_names[i].split('.')[0].split('_')
    if len(split) > 2:
        contrast = split[2]
        if len(split) > 3:
            way = split[3]
            if len(split) > 4:
                K = split[4]
            else:
                K = None
        else:
            way = None
    else:
        contrast = None
    print(file.shape)
    print(anat_map_array.shape)
    nX, nY, nZ = file.shape
    print('file_name =', array_names[i])
    _ = plot_maps(None, None, array_to_show=file, save=False, mode='other', contrast_name=contrast, \
                  way=way, K=K, anat_map_array=anat_map_array, nX=nX, nY=nY, nZ=nZ)
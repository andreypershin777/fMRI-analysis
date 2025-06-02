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
voxel_coord, coord_voxel, load_patient_logs, print_matrix, print_vector, get_Y, NpEncoder, make_data_for_SPM

patients_folder = ''
#patients = []
#subs = []
save_path = ''
patients = ['M_Y3_003', 'M_Y3_007', 'M_Y3_012', 'M_Y3_019', 'M_Y3_024', 'M_Y3_025', \
               'M_Y3_040', 'M_Y3_042', 'M_Y3_046', 'M_Y3_057']
subs = ['sub-03', 'sub-07', 'sub-12', 'sub-19', 'sub-24', 'sub-25', 'sub-40', 'sub-42', 'sub-46', 'sub-57']
runs = ['me_1', 'me_2', 'me_3', 'fr_1', 'fr_2', 'fr_3']
new_patient_path = r'C:\Users\andrew\Desktop\Mag Diploma\Код и новые данные\SPM'
patient = patients[0]
run = runs[0]
patient_data_folder = r'C:\Users\andrew\Desktop\Mag Diploma\Код и новые данные'
patient_logs_folder = r'C:\Users\andrew\Desktop\Mag Diploma\Код и новые данные'
for i in range(len(patients)):
    patient = patients[i]
    sub = subs[i]
    patient_data_path = os.path.join(patient_data_folder, patient)
    patient_logs_path = os.path.join(patient_logs_folder, patient, 'log')
    make_data_for_SPM(patient_data_path, patient_logs_path, sub, new_patient_path, runs)
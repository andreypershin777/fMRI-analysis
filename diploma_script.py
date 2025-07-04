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
from nilearn.masking import compute_brain_mask
import pickle
from diploma_functions import get_X, load_patient_data, chunk_voxels, run_computing_on_chunk,\
computing, get_best_thresholds, get_reproducibility, get_map_array, plot_maps
from diploma_functions import is_serializable, total_size, coord_voxel, f_obj, get_thresholds_dict, get_t_groups, \
voxel_coord, coord_voxel, load_patient_logs, print_matrix, print_vector, get_Y, NpEncoder, smooth_covariance_maps, \
get_t_value, get_T_value, Tee
from nilearn.image import resample_img


def main():
    patient_data_path = r'C:\Users\andrew\Desktop\Mag Diploma\Code_and_new_data\data_preprocessed\M_Y3_057'
    patient_logs_path = os.path.join(patient_data_path, 'log')
    patient = os.path.basename(patient_data_path)
    save_path = r'C:\Users\andrew\Desktop\Mag Diploma\Code_and_new_data'
    nX = 53
    nY = 63
    nZ = 52
    #nX = 25
    #nY = 25
    #nZ = 25
    nV = nX*nY*nZ
    nT = 72
    nS = 18
    nR = 6
    nP = 5
    
    target_shape = (nX, nY, nZ)
    scans = os.listdir(os.path.join(patient_data_path, 'me_1'))
    first_scan_path = os.path.join(patient_data_path, 'me_1', scans[0])
    first_scan = nib.load(first_scan_path)
    brain_mask = compute_brain_mask(first_scan, threshold=0.1)
    mask_data = brain_mask.get_fdata()
    voxels = []
    for x in range(nX):
        for y in range(nY):
            for z in range(nZ):
                if mask_data[x, y, z] != 0:
                    voxels.append(coord_voxel(x, y, z, nX, nY, nZ))

    print('Количество вокселей для анализа:', len(voxels))
    print('Общее количество вокселей:', nV)

    voxels_try_array = np.zeros((nX, nY, nZ))
    for v in voxels:
        x, y, z = voxel_coord(v, nX, nY, nZ)
        voxels_try_array[x, y, z] = 1
    #plt.imshow(voxels_try_array[:, :, nZ // 2], cmap='gray', interpolation='nearest')
    #plt.show()

    #anat_scan_array = nib.load(os.path.join(patient_data_path, 'structural', 'wm' + patient + '_T1.nii')).get_fdata()
    #arr1 = anat_scan_array[:, :, anat_scan_array.shape[2] // 2]
    #plt.imshow(arr1, cmap='gray', interpolation='nearest')
    #plt.show()
    #print('Размерность anat_scan_array:', anat_scan_array.shape)
    #print('Размерность voxels_try_array:', voxels_try_array.shape)
    #_ = plot_maps(None, None, array_to_show=voxels_try_array, save=False, mode='try', \
    #              contrast_name='try', way='try', K=0, anat_map_array=anat_scan_array, \
    #                nX=nX, nY=nY, nZ=nZ)
    
    k = nS
    k2 = 2
    X2 = [[1,1],[1,1],[1,1],[1,-1],[1,-1],[1,-1]]
    X2 = np.array(X2)
    
    #contrast = np.array([1 if i%2==0 else -1 for i in range(nS)]) #Пока взяли 'какой-то контраст'
    
    K_list = [30, 40, 50, 60]
    
    #x_list = [3, 4]
    x_list = []
    
    #y_list = [3, 4]
    y_list = []
    
    #z_list = [0, 1]
    z_list = []

    full_data = False

    mode = 'numb'

    ways = ['other', 'lamda']
    #way = 'lamda'

    lamda = 0.21

    auto_end = True

    max_iter = 100

    one_scan_time = 2.5

    need_gauss_reg = True

    frametimes = []
    slicetimes = []
    events = []

    n_slices = 25
    one_slice_time = 0.1
    
    #############################################################################
    
    t0 = time.time()
    start = time.time()
    # узнаем количество ядер у процессора
    #n_proc = multiprocessing.cpu_count()
    #procs = n_proc
    # вычисляем сколько циклов вычислений будет приходится
    # на 1 ядро, что бы в сумме получилось 80 или чуть больше
    #voxels = [i for i in range(nV)]
    #calcs = [[] for _ in range(n_proc)]
    #for i in range(n_proc):
    #    calcs[i] = [v for v in voxels if v % n_proc == i]
    #print('Начали мультипроцессинг')
    #Start
    #print(n_proc, len(calcs))
    
    logs_dict = load_patient_logs(patient_logs_path)
    runs = list(logs_dict.keys())

    log = logs_dict[runs[0]]
    #log = log[log.index % 2 == 0]
    #print(log)
    frametimes_0 = [one_scan_time*i for i in range(nT)]
    

    #slicetimes_0 = [(i//2+1) * one_slice_time if i % 2 == 0 else ((n_slices+1) // 2 + (i+1) // 2) * one_slice_time for i in range(n_slices)]

    #slicetimes_0 = [(i+1)*one_slice_time for i in range(n_slices)]
    slicetimes_0 = [0.1 for i in range(nZ)]

    change_dict = {'FrMe_06.png': 'MeFr_04.png', 'FrMe_08.png': 'MeFr_02.png', 'FrMe_10.png': 'MeFr_00.png', \
                   'FrMe_00.png': 'MeFr_10.png', 'FrMe_04.png': 'MeFr_06.png', 'FrMe_02.png': 'MeFr_08.png'}

    frametimes = {}
    slicetimes = {}
    for run in runs:
        frametimes[run]  = frametimes_0.copy()
        slicetimes[run] = slicetimes_0.copy()
    
    patient_data_dict = load_patient_data(patient_data_path)

    for run in runs:
        for ind in range(0, len(logs_dict[run])-1, 2):
            logs_dict[run].loc[ind, 'values.ResponseForSubj'] = logs_dict[run].loc[ind+1, 'values.ResponseForSubj']
            #print(logs_dict[run].drop(columns=['date', 'time', 'values.TestBlockNum', 'response', 'values.WaitedForSynchro', 'latency', 'blockcode', 'values.PreStimInterval']))
        logs_dict[run] = logs_dict[run][(logs_dict[run].index % 2 == 0)  & \
                                        (logs_dict[run]['values.TestTrialNum'] != 0)]
        #logs_dict[run] = logs_dict[run].reset_index(drop=True)
        #logs_dict[run] = logs_dict[run].drop(columns=['date', 'time', 'blockcode', 'values.WaitedForSynchro', 'response', 'values.ResponseForSubj'])
        #print(logs_dict[run])
        #print('run =', run)
        #print(logs_dict[run]['stimulusitem1'].value_counts())

    if full_data:
        nT = 144
        nS = 54
        for run in runs:
            images = logs_dict[run]['stimulusitem1'].values
            for i in range(len(images)):
                for key in change_dict.keys():
                    if images[i] == key:
                        images[i] = change_dict[key]
                        #print('Изменили', key, 'на', change_dict[key], 'в', run, 'в позиции', i)
                        break
            logs_dict[run].loc[:,'stimulusitem1'] = images

        for run in runs:
            images = logs_dict[run]['stimulusitem1'].values
            for i in range(len(images)):
                image = images[i].split('.')
                trial_num = logs_dict[run]['values.TestTrialNum'].values[i]
                n_trial = (trial_num - 1) // 12 + 1
                new_part = '_' + str(n_trial)
                images[i] = image[0] + new_part + '.' + image[1]

        #logs_dict_common = logs_dict.copy()
        #for run in runs:
        #    logs_dict_common[run] = logs_dict_common[run][(logs_dict[run]['stimulusitem1'].str.contains('MeFr') | \
        #                                        logs_dict[run]['stimulusitem1'].str.contains('FrMe'))]

        images_order = []
        for run in runs:
            log = logs_dict[run]
            images = log['stimulusitem1'].values
            for image in images:
                if image not in images_order:
                    images_order.append(image)
        #images_order = [images[i] for i in range(0, len(images)-1, 2)]
        #print(len(images_order), 'изображений в логе')
        #print('images_order:\n', images_order)
        contrast_me = [0 for i in range(nS)]
        contrast_fr = [0 for i in range(nS)]
        
        #target_morf_list = [0 for _ in range(len(images_order))]
        #response_diff_list = [0 for _ in range(len(images_order))]
        me_morf_list = []
        fr_morf_list = []
        median_morf_me = 0.5
        median_morf_fr = 0.5
        for run in runs:
            log = logs_dict[run]
            for i, image in enumerate(images_order):
                image_slice = log[log['stimulusitem1'] == image]
                if len(image_slice) > 0:
                    me_morf = image_slice['values.MeMorphShare'].values[0] / 100
                    fr_morf = image_slice['values.FrMorphShare'].values[0] / 100
                    me_morf_list.append(me_morf)
                    fr_morf_list.append(fr_morf)
                    #response = image_slice['values.ResponseForSubj'].values[0] / 100
                else:
                    continue
                #response_diff = abs(response - target_morf)
                if not (('MeFr' in image) or ('FrMe' in image)):
                    continue
                #print(i, image, len(contrast_me), len(contrast_fr))
                if ('MeFr' in image) or ('FrMe' in image):
                    contrast_me[i] = me_morf - 0.5
                    contrast_fr[i] = fr_morf - 0.5
                else:
                    contrast_me[i] = 0
                    contrast_fr[i] = 0
                #contrast_me[i] = (1 if me_morf > median_morf_me else -1)
                #contrast_fr[i] = (1 if fr_morf > median_morf_fr else -1)
                #print(f'{contrast[i]} = (1 - {response_diff}) * {target_morf}')
                #target_morf_list[i] = target_morf
                #response_diff_list[i] = response_diff
        contrasts = {'me': contrast_me}

        #for i, image in enumerate(images_order):
        #    print(image, '-', contrast_me[i], '-', me_morf_list[i], '-', fr_morf_list[i])
        #    print(image, '-', contrast_fr[i], '-', me_morf_list[i], '-', fr_morf_list[i])

        #print(len(images_order))
        #print(images_order)
        #print(len(logs_dict[runs[0]]))
        #print(logs_dict[runs[0]]['stimulusitem1'].values)


        

        '''
        for run in runs:
            responses_dict = {}
            log = logs_dict[run]
            for i, image in enumerate(images_order):
                image_slice = log[log['stimulusitem1'] == image]
                if len(image_slice) > 0:
                    response = image_slice['values.ResponseForSubj'].values[0]
                    responses_dict[image] = response
            median_response = np.median(list(responses_dict.values()))

            for i, image in enumerate(images_order):
                if image in responses_dict.keys():
                    if responses_dict[image] >= median_response:
                        contrast[i] = 1
                    else:
                        contrast[i] = -1
                else:
                    contrast[i] = 0
        '''

        

        events = {}
        #ids = [i//2+1 for i in range(0, len(log)-1, 2)]

        for r, run in enumerate(runs):
            log = logs_dict[run]
            images_r = log['stimulusitem1'].values
            #images_r = [images[i] for i in range(0, len(images)-1, 2)]
            events[run] = np.zeros((4, nS))
            gap_r = log['values.PreStimInterval'].values
            #gap_r = [gap[i] for i in range(0, len(gap)-1, 2)]
            for s in range(len(images_r)):
                real_n = images_order.index(images_r[s])
                #print(images_r[s], s, real_n)
                gap = gap_r[s]
                events[run][0][real_n] = real_n + 1
                events[run][1][real_n] = 10 + s * 10 + gap/1000
                events[run][2][real_n] = 2.5
                events[run][3][real_n] = 1
            for s in range(len(images_order)):
                if events[run][0][s] == 0:
                    events[run][0][s] = s + 1
                    events[run][1][s] = -1
                    events[run][2][s] = -1
                    events[run][3][s] = 0
    else:
        nT = 72
        nS = 18
        necessary_scans = {}
        for run in runs:
            images = logs_dict[run]['stimulusitem1'].values
            for i in range(len(images)):
                for key in change_dict.keys():
                    if images[i] == key:
                        images[i] = change_dict[key]
                        #print('Изменили', key, 'на', change_dict[key], 'в', run, 'в позиции', i)
                        break
            logs_dict[run].loc[:,'stimulusitem1'] = images


        for run in runs:
            images = logs_dict[run]['stimulusitem1'].values
            for i in range(len(images)):
                image = images[i].split('.')
                trial_num = logs_dict[run]['values.TestTrialNum'].values[i]
                n_trial = (trial_num - 1) // 12 + 1
                new_part = '_' + str(n_trial)
                images[i] = image[0] + new_part + '.' + image[1]

        #logs_dict_common = logs_dict.copy()
        #for run in runs:
        #    logs_dict_common[run] = logs_dict_common[run][(logs_dict[run]['stimulusitem1'].str.contains('MeFr') | \
        #                                        logs_dict[run]['stimulusitem1'].str.contains('FrMe'))]

        for run in runs:
            logs_dict[run] = logs_dict[run].loc[(logs_dict[run]['stimulusitem1'].str.contains('MeFr')) | \
                                    (logs_dict[run]['stimulusitem1'].str.contains('FrMe'))]
        
        images_order = []
        for run in runs:
            log = logs_dict[run]
            images = log['stimulusitem1'].values
            for image in images:
                if image not in images_order:
                    images_order.append(image)
        
        for run in runs:
            necessary_scans[run] = []
            log = logs_dict[run]
            images = log['stimulusitem1'].values
            for image in images:
                trial_number = log.loc[log['stimulusitem1'] == image, 'values.TestTrialNum'].values[0]
                scans_numbers = [i for i in range(trial_number*4, trial_number*4 + 4)]
                necessary_scans[run].extend(scans_numbers)
                del scans_numbers
        
        new_patient_data_dict = {}
        for run in runs:
            new_patient_data_dict[run] = []
            for scan_number in necessary_scans[run]:
                new_patient_data_dict[run].append(patient_data_dict[run][scan_number])
        patient_data_dict = new_patient_data_dict.copy()

        #for run in runs:
        #    print(logs_dict[run]['values.TestTrialNum'].values)
        #    print(necessary_scans[run])

        #images_order = [images[i] for i in range(0, len(images)-1, 2)]
        #print(len(images_order), 'изображений в логе')
        #print('images_order:\n', images_order)
        contrast_me = [0 for i in range(nS)]
        contrast_fr = [0 for i in range(nS)]
        
        #target_morf_list = [0 for _ in range(len(images_order))]
        #response_diff_list = [0 for _ in range(len(images_order))]
        me_morf_list = []
        fr_morf_list = []
        median_morf_me = 0.5
        median_morf_fr = 0.5
        for run in runs:
            log = logs_dict[run]
            for i, image in enumerate(images_order):
                image_slice = log[log['stimulusitem1'] == image]
                if len(image_slice) > 0:
                    me_morf = image_slice['values.MeMorphShare'].values[0] / 100
                    fr_morf = image_slice['values.FrMorphShare'].values[0] / 100
                    me_morf_list.append(me_morf)
                    fr_morf_list.append(fr_morf)
                    #response = image_slice['values.ResponseForSubj'].values[0] / 100
                else:
                    continue
                #response_diff = abs(response - target_morf)
                if not (('MeFr' in image) or ('FrMe' in image)):
                    continue
                #print(i, image, len(contrast_me), len(contrast_fr))
                contrast_me[i] = (1 if me_morf > median_morf_me else -1)
                contrast_fr[i] = (1 if fr_morf > median_morf_fr else -1)
                #print(f'{contrast[i]} = (1 - {response_diff}) * {target_morf}')
                #target_morf_list[i] = target_morf
                #response_diff_list[i] = response_diff
        contrasts = {'me': contrast_me}

        #for i, image in enumerate(images_order):
        #    print(image, '-', contrast_me[i], '-', me_morf_list[i], '-', fr_morf_list[i])
        #    print(image, '-', contrast_fr[i], '-', me_morf_list[i], '-', fr_morf_list[i])

        #print(len(images_order))
        #print(images_order)
        #print(len(logs_dict[runs[0]]))
        #print(logs_dict[runs[0]]['stimulusitem1'].values)


        

        '''
        for run in runs:
            responses_dict = {}
            log = logs_dict[run]
            for i, image in enumerate(images_order):
                image_slice = log[log['stimulusitem1'] == image]
                if len(image_slice) > 0:
                    response = image_slice['values.ResponseForSubj'].values[0]
                    responses_dict[image] = response
            median_response = np.median(list(responses_dict.values()))

            for i, image in enumerate(images_order):
                if image in responses_dict.keys():
                    if responses_dict[image] >= median_response:
                        contrast[i] = 1
                    else:
                        contrast[i] = -1
                else:
                    contrast[i] = 0
        '''

        

        events = {}
        #ids = [i//2+1 for i in range(0, len(log)-1, 2)]

        for r, run in enumerate(runs):
            log = logs_dict[run]
            images_r = log['stimulusitem1'].values
            #images_r = [images[i] for i in range(0, len(images)-1, 2)]
            events[run] = np.zeros((4, nS))
            gap_r = log['values.PreStimInterval'].values
            #gap_r = [gap[i] for i in range(0, len(gap)-1, 2)]
            for s in range(len(images_r)):
                real_n = images_order.index(images_r[s])
                #print(images_r[s], s, real_n)
                gap = gap_r[s]
                events[run][0][real_n] = real_n + 1
                events[run][1][real_n] = s * 10 + gap/1000
                events[run][2][real_n] = 2.5
                events[run][3][real_n] = 1
            for s in range(len(images_order)):
                if events[run][0][s] == 0:
                    events[run][0][s] = s + 1
                    events[run][1][s] = -1
                    events[run][2][s] = -1
                    events[run][3][s] = 0

    #events[0] = ids
    #duration = list(log['latency'])
    #for i in range(len(duration)):
    #    duration[i] = duration[i] / 1000
    #events[1] = [5 + 2.5 * 4 * i for i in range(0, (len(log)-1)//2)]
    #events[2] = [duration[i] for i in range(0, len(log)-1, 2)]
    #events[3] = [1 for i in range(0, len(log)-1, 2)]
    #events = np.array(events).T
    
    #print('frametimes:\n', frametimes)
    #print('slicetimes:\n', slicetimes)

    #print('events:\n', events)

    voxel1 = (5, 6, 7)
    voxel2 = (4, 5, 6)
    Y_list1 = get_Y(patient_data_path, voxel1, runs=runs, prew=False)
    Y_list2 = get_Y(patient_data_path, voxel2, runs=runs, prew=False)
    #print(runs)
    #for i, run in enumerate(runs):
    #    #print_vector(Y_list1[i], 'Y_list1[{}]'.format(run))
    #    print('1, ', i, Y_list1[i].shape, '\n', Y_list1[i])
    #    #print_vector(Y_list2[i], 'Y_list2[{}]'.format(run))
    #    print('2, ', i, Y_list2[i].shape, '\n', Y_list2[i])
    #    print()


    list_dir = os.listdir(path=save_path)
    for i in range(10000):
        folder_name = 'Запуск_' + str(i)
        if folder_name in list_dir:
            continue
        else:
            n_try = i
            break
    os.chdir(save_path)
    save_path = os.path.join(save_path, folder_name)
    os.mkdir(folder_name)
    os.chdir(save_path)


    output_path = os.path.join(save_path, 'output.txt')
    f = open(output_path, 'w')
    sys.stdout = Tee(sys.stdout, f)

    #print("Hello, world!")
    #print("Это будет и в консоли, и в файле.")


    z_slices = [i for i in range(nZ)]
    X_matrixes = []
    print('Вычисляем матрицы X:')
    t1 = time.time()
    for z in z_slices:
        #print('z =', z)
        X_list = get_X(patient_logs_path, (0, 0, z),  nX, nY, nZ, nR, nT, nS, runs=runs, \
                       frametimes=frametimes, slicetimes=slicetimes, events=events);
        X_matrixes.append(X_list.copy())

    
    #for r, run in enumerate(runs):
    #    for z in range(len(X_matrixes)):
    #        print(f'z = {z}')
    #        print(events[run][0])
    #        print(events[run][1])
    #        print_matrix(X_matrixes[z][r], 'X_matrixes[{}][{}]'.format(z, r), events[run][1])

    for i in range(len(X_matrixes)):
        for x in range(len(X_matrixes[i])):
            for y in range(len(X_matrixes[i][x])):
                for z in range(len(X_matrixes[i][x, y])):
                    if X_matrixes[i][x, y, z] is np.nan:
                        print(X_matrixes[i][x, y, z], i, x, y, z)
    t2 = time.time()
    print('Затрачено времени:', t2-t1)
    
    print('Загружаем данные Y:')
    t3 = time.time()
    #patient_data_dict = load_patient_data(patient_data_path)
    t4 = time.time()
    print('Загрузили за', t4-t3)
    
    
    print('Начали мультипроцессинг')
    t1 = time.time()
    
    
    ##################################################################################
    
    num_workers = os.cpu_count()
    print(f"num_workers = {num_workers}")
    voxel_chunks = chunk_voxels(voxels, num_workers, max_iter)
    
    args_list = [
        (chunk, X_matrixes, patient_data_dict, contrasts, K_list, \
         X2, x_list, y_list, z_list, nX, nY, nZ, nR, nT, nS, max_iter, runs)   
        for chunk in voxel_chunks
    ]
    
    ful_args = (voxels, X_matrixes, patient_data_dict, contrasts, K_list, \
         X2, x_list, y_list, z_list, nX, nY, nZ, nR, nT, nS, max_iter, runs)
    
    t_values_final = {}
    T_values_final = {}
    mean_beta_final = {}
    cov_beta_final = {}
    Omega_final = {}
    mu_final = {}
    for contrast_name in contrasts.keys():
        t_values_final[contrast_name] = {}
        T_values_final[contrast_name] = {}
    
    for i, args in enumerate(args_list):
        try:
            pickle.dumps(args)
        except Exception as e:
            print(f"args_list[{i}] не сериализуем: {e}")
        else:
            print(f"args_list[{i}] сериализуем")
        #print(is_serializable(arg))
        #print(total_size(arg))
        #print()
    
    result_files = []
    print('Начало процессов')
    

    if(len(voxels) > 500):
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = executor.map(run_computing_on_chunk, args_list)

            for t_chunk, T_chunk, mean_beta_chunk, cov_beta_chunk, Omega_chunk, mu_chunk in results:
                for contrast_name in contrasts.keys():
                    t_values_final[contrast_name].update(t_chunk[contrast_name])
                    T_values_final[contrast_name].update(T_chunk[contrast_name])
                    mean_beta_final.update(mean_beta_chunk)
                    cov_beta_final.update(cov_beta_chunk)
                    Omega_final.update(Omega_chunk)
                    mu_final.update(mu_chunk)
    else:
        t_values_final, T_values_final, mean_beta_final, cov_beta_final, Omega_final, mu_final\
              = run_computing_on_chunk(ful_args)
        
    #print(mean_beta_final.keys())

    if need_gauss_reg:
        cov_matrixes = np.zeros((nX, nY, nZ, nS, nS))
        for r in range(nR):
            for x in range(nX):
                for y in range(nY):
                    for z in range(nZ):
                        if coord_voxel(x, y, z, nX, nY, nZ) not in cov_beta_final.keys():
                            continue
                        cov_matrixes[x, y, z, :, :] = cov_beta_final[coord_voxel(x, y, z, nX, nY, nZ)][r, :, :]
            smoothed_cov_matrixes = smooth_covariance_maps(cov_matrixes)
            for x in range(nX):
                for y in range(nY):
                    for z in range(nZ):
                        if coord_voxel(x, y, z, nX, nY, nZ) not in cov_beta_final.keys():
                            continue
                        cov_beta_final[coord_voxel(x, y, z, nX, nY, nZ)][r, :, :] = smoothed_cov_matrixes[x, y, z, :, :]
        for x in range(nX):
            for y in range(nY):
                for z in range(nZ):
                    v = coord_voxel(x, y, z, nX, nY, nZ)
                    if v in voxels:
                        mean_beta = mean_beta_final[v]
                        disp_beta = cov_beta_final[v]
                        Omega = Omega_final[v]
                        mu = mu_final[v]
                        for contrast_name, contrast in contrasts.items():
                            t_value_list = get_t_value(mean_beta, disp_beta, contrast, nX, nY, nZ, nR, nT, nS)
                            T_value = get_T_value(mu, Omega, contrast, nX, nY, nZ, nR, nT, nS)
                            t_values_final[contrast_name][v] = np.array(t_value_list)
                            T_values_final[contrast_name][v] = T_value
                            del t_value_list
                            del T_value
    
    '''
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        #results = executor.map(run_computing_on_chunk, args_list)
        futures = [executor.submit(run_computing_on_chunk, args) for args in args_list]
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                result_files.append(result)
            except Exception as e:
                print(f"Ошибка в процессе {e}")
    #print()
    #print(is_serializable(results), results)
    #for t_chunk, T_chunk in results:
    #    t_values_final.update(t_chunk)
    #    T_values_final.updata(T_chunk)

    for file in result_files:
        data = np.load(file, allow_pickle=True)
        t_values = data['t_values'].item()
        T_values = data['T_values'].item()
        t_values_final.update(t_values)
        T_values_final.update(T_values)
        os.remove(file)
    '''
    #for contrast_name in contrasts.keys():
    #    print(contrast_name, len(t_values_final[contrast_name]), len(T_values_final[contrast_name]))
    
    #time.sleep(5)
    #print("t_values_final:", t_values_final)
    #print("T_values_final:", T_values_final)
    #t_values_gen = {}
    #T_values_gen = {}
    t_values_gen = t_values_final.copy()
    T_values_gen = T_values_final.copy()
    voxel_status_dict_gen = {}
    '''
    for contrast_name in contrasts.keys():
        t_values_gen[contrast_name] = np.zeros((len(t_values_final[contrast_name]), nR))
        T_values_gen[contrast_name] = np.zeros((len(T_values_final[contrast_name]), nR))
    
    for contrast_name in contrasts.keys():
        for i, (key, value) in enumerate(t_values_final[contrast_name].items()):
            t_values_gen[contrast_name][key, :] = value
            T_values_gen[contrast_name][key] = T_values_final[contrast_name][key]
    '''
    #for contrast_name in contrasts.keys():
    #    print(contrast_name, len(t_values_gen[contrast_name]), len(T_values_gen[contrast_name]))
    
    #time.sleep(5)
    #t_values_gen = t_values_final.copy()
    #T_values_gen = T_values_final.copy()
    
    final_thresholds_gen = {}
    for contrast_name in contrasts.keys():
        final_thresholds_gen[contrast_name] = {}
        for way in ways:
            final_thresholds_gen[contrast_name][way] = {}


    for contrast_name, contrast in contrasts.items():
        for way in ways:
            print('__________', 'contrast =', contrast_name, '___', 'way =', way, '___________')
            t_values = t_values_gen[contrast_name]
            T_values = T_values_gen[contrast_name]
            print('Размер t_values:', len(t_values), len(t_values[coord_voxel(nX//2, nY//2, nZ//2, nX, nY, nZ)]))
            dict_thresholds = get_thresholds_dict(K_list, t_values, nX, nY, nZ, nR, nT, nS, mode)
            print(dict_thresholds)
            t_values_voxel_groups = get_t_groups(K_list, dict_thresholds, t_values, nX, nY, nZ, nR, nT, nS, mode)
            for K in K_list:
                print('K =', K)
                print('t_values')
                for v in t_values.keys():
                    if any(t_values[v] == np.nan):
                        print('t_values[v] =', t_values[v])
                    for i in range(len(t_values[v])):
                        if t_values[v][i] == np.nan:
                            print(v, i)
                            print('t_values[v][i] =', t_values[v][i])
                print('t_values_groups_counts')
                sum_counts = [0] * (K + 1)
                for v in t_values_voxel_groups[K].keys():
                    for i in range(len(t_values_voxel_groups[K][v])):
                        sum_counts[i] += t_values_voxel_groups[K][v][i]
                print(sum_counts)
                #print(np.sum(t_values_voxel_groups[K], axis=0))
                #print(np.sum(np.sum(t_values_voxel_groups[K], axis=0), axis=0))
                print(len(voxels) * nR)
                #for v in range(t_values_voxel_groups[K].shape[0]):
                #    for i in range(t_values_voxel_groups[K].shape[1]):
                #        print(t_values_voxel_groups[K][v][i], end=' ')
                #    print()



            n_t_dict = {}
            for v in t_values.keys():
                t = tuple(t_values[v])
                if t not in n_t_dict.keys():
                    n_t_dict[t] = 0
                n_t_dict[t] += 1
            print('Число уникальных наборов t_values:', len(n_t_dict.keys()))
            #print(n_t_dict.keys())
            #print(n_t_dict)
            t1 = time.time()
            print('Мультипроцессинг за', t1-t0)
            final_thresholds, P_A_dict, P_I_dict, lamda_dict = get_best_thresholds(K_list, t_values, nX, \
                                                        nY, nZ, nR, nT, nS, mode, way=way, contrast_name=contrast_name, \
                                                            save_path=save_path, need_plot=False, lamda=lamda)
            lamda_0 = 0.02
            dict_thresholds = get_thresholds_dict(K_list, t_values, nX, nY, nZ, nR, nT, nS, mode)
            t_values_voxel_groups = get_t_groups(K_list, dict_thresholds, t_values, nX, nY, nZ, nR, nT, nS, mode)
            t_values_voxel_groups_np = {}
            for K in K_list:
                t_values_voxel_groups_np[K] = np.zeros((len(t_values_voxel_groups[K].keys()), K+1))
                for i, v in enumerate(t_values_voxel_groups[K].keys()):
                    t_values_voxel_groups_np[K][i, :] = t_values_voxel_groups[K][v]
            for K in K_list:
                #print('t_values')
                #print(t_values)
                #print('t_values_groups_counts')
                #print(np.sum(t_values_voxel_groups[K], axis=0))
                #print('t_values_voxel_groups')
                #for v in range(t_values_voxel_groups[K].shape[0]):
                #    print(t_values_voxel_groups[K][v])
                #print(t_values_voxel_groups[K])
                P_A_list0 = np.array([1/(K+1) for i in range(K+1)])
                P_I_list0 = np.array([1/(K+1) for i in range(K+1)])
                print('lamda_0 =', lamda_0)
                print('P_A_list_0 =', P_A_list0)
                print('P_I_list_0 =', P_I_list0)
                print('Исходное значение функции правдоподобия :\n', f_obj(lamda_0, P_A_list0, \
                                                                        P_I_list0, t_values_voxel_groups_np[K]))
                print('lamda =', lamda_dict[K])
                print('P_A_list =', P_A_dict[K])
                print('P_I_list =', P_I_dict[K])
                print('Итоговое значение функции правдоподобия :\n', f_obj(lamda_dict[K], P_A_dict[K], \
                                                                        P_I_dict[K], t_values_voxel_groups_np[K]))
            print(f'contrast = {contrast_name}')
            print(f'way = {way}')
            print('Final thresholds:', final_thresholds)
            t2 = time.time()
            print('Лучший порог за', t2-t1)
            final_thresholds_gen[contrast_name][way] = final_thresholds.copy()

    voxel_status_dict_gen = {}
    for contrast_name in contrasts.keys():
        voxel_status_dict_gen[contrast_name] = {}
        for way in ways:
            voxel_status_dict_gen[contrast_name][way] = {}

    print(final_thresholds_gen)

    for contrast_name in contrasts.keys():
        for way in ways:
            t2 = time.time()
            voxel_activity_dict = get_reproducibility(K_list, final_thresholds_gen[contrast_name][way], \
                                                        t_values_gen[contrast_name], nX, nY, nZ, nR, nT, nS, mode)
            activity = {}
            for K in K_list:
                print('K =', K)
                print('sum =', np.sum(voxel_activity_dict[K], axis=0))
                #print('voxel_activity_dict:')
                #print(voxel_activity_dict)
                for v in range(voxel_activity_dict[K].shape[0]):
                    #if voxel_activity_dict[K][v] > 0:
                    #print(voxel_coord(v, nX, nY, nZ), voxel_activity_dict[K][v])
                    if voxel_activity_dict[K][v] not in activity.keys():
                        activity[voxel_activity_dict[K][v]] = []
                    #print('voxel_activity_dict[K][v] =', voxel_activity_dict[K][v], 'для', voxel_coord(v, nX, nY, nZ))
                    activity[voxel_activity_dict[K][v]].append(voxel_coord(v, nX, nY, nZ))
                for a in activity.keys():
                    print('Активность', a, 'в', len(activity[a]), 'вокселях')
                    #for v in activity[a]:
                    #    print(v, end=' ')
                    print()
            voxel_status_dict, _ = get_map_array(K_list, voxel_activity_dict, nX, nY, nZ, nR, nT, nS, mode, strong_threshold=0.8, normal_threshold=0.6)
            #final_voxel_status_dict = {}
            voxel_status_dict_gen[contrast_name][way] = voxel_status_dict.copy()
            #print('voxel_activity_dict:')
            #print(voxel_activity_dict)
            #print('voxel_status_dict:')
            #print(voxel_status_dict)
            t3 = time.time()
            print('Воспроизводимость за', t3-t2)

    if (x_list != []) or (y_list != []) or (z_list != []):
        print('Рисуем для:')
        print(x_list)
        print(y_list)
        print(z_list)

    final_voxel_status_dict_gen = {}
    for contrast_name in contrasts.keys():
        final_voxel_status_dict_gen[contrast_name] = {}
        for way in ways:
            final_voxel_status_dict_gen[contrast_name][way] = {}

    n_try = 0
    for contrast_name, contrast in contrasts.items():
        for way in ways:
            t_values = t_values_gen[contrast_name]
            for K in K_list:
                print()
                print(f'Построение графиков для K={K} порогов, contrast = {contrast_name}, way = {way}')

                final_voxel_status_dict_gen[contrast_name][way][K] = plot_maps(voxel_status_dict_gen[contrast_name][way][K], t_values, x_list, y_list, z_list, nX, nY, nZ, \
                                                    nR, nT, nS, contrast_name==contrast_name, mode='str', auto_end=auto_end)
                
            parameters_dict = {'nX': nX, 'nY': nY, 'nZ': nZ, 'nS': nS, 'nR': nR, 'nT' : nT, \
                            'mode': mode, 'way': way, 'K': K_list, 'contrast': contrast, 'X*': X2, 'max_iter': max_iter, \
                        'data_path': patient_data_path, 'logs_path': patient_logs_path, 'one_scan_time': one_scan_time, 'n_slices': n_slices, \
                                'runs': runs}
            
            for key in parameters_dict.keys():
                value = parameters_dict[key]
                if not is_serializable(value):
                    print(key)
            
            json_name = 'parameters_' + contrast_name + '_' + way + '.json'
            json_path = os.path.join(save_path, json_name)
            with open(json_path, 'w') as params:
                json.dump(parameters_dict, params, cls=NpEncoder)

            for K in K_list:
                file_name = 'reproducibility_array_' + contrast_name + '_' + way + '_' + str(K) + '.npy'
                file_path = os.path.join(save_path, file_name)
                np.save(file_path, final_voxel_status_dict_gen[contrast_name][way][K])




if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
    
    
    
    
    
    
    
    

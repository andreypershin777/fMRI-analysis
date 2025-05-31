import pandas as pd
#with warnings.catch_warnings():
#    warnings.filterwarnings("ignore", message=".*'cgi' is deprecated.*", category=DeprecationWarning)
#    import mne
import copy
import random
import time
import math
import numpy as np
import os
import json
import sys
import multiprocessing
from multiprocessing import Process, Queue, Pool
import matplotlib.pyplot as plt
#%matplotlib inline
import nibabel as nib # common way of importing nibabel
import scipy.optimize as scopt
import numpy as np
import scipy
import scipy.stats as sps
import traceback
from scipy.optimize import LinearConstraint
from scipy.linalg import inv, solve, pinv
from scipy.sparse import csr_matrix, issparse
from scipy.interpolate import UnivariateSpline
from scipy import interpolate
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.linalg import svd, pinv
from scipy.optimize import SR1
from scipy.sparse.linalg import spsolve
from sklearn.linear_model import Ridge
from scipy import integrate
import concurrent.futures
from typing import List, Dict, Tuple
import pickle
from scipy.integrate import IntegrationWarning
from scipy.special import logsumexp
import warnings

def load_data(data_link, numb_patients):
    
    data_dict = {}
    os.chdir(data_link)
    list_dir = os.listdir()
    i = 0
    for directory in list_dir:
        if directory == 'M_y2_002': #пропускаем, вроде были проблемы
            continue
        if i >= numb_patients: #берём данные о первых пациентах
            break
        
        #Пациент
        patient = directory.split('_')[2:] #номер пациента
        if len(patient) == 1:
            patient = patient[0]
        else:
            patient = patient[0] + '_' + patient[1]
        patient_path = os.path.join(data_link, directory)
        os.chdir(patient_path)
        
        #Прогоны для одного пациента
        runs = os.listdir()
        for run in runs:
            #Конкретный прогон
            run_path = os.path.join(patient_path, run)
            
            #Проверка, что это одна из нужных папок папка 
            if run.find('.') == -1 and len(run.split('_')) > 1 \
            and run.split('_')[1].isnumeric():
                
                #Список сканов
                scans = os.listdir(path=run_path)
                for scan in scans:
                    
                    #Конкретный скан
                    scan_path = os.path.join(run_path, scan)
                    
                    #Проверка, что это действительно файл фМРТ
                    if scan.find('.nii') != -1:
                        if patient not in data_dict.keys():
                            data_dict[patient] = {}
                        if run not in data_dict[patient].keys():
                            data_dict[patient][run] = []
                        scan_img = nib.load(scan_path)
                        data_dict[patient][run].append(scan_img.get_fdata())
        i = i + 1
    return data_dict


def load_run_data(run_data_path):
    scans_list = []
    
    scans = os.listdir(path=run_data_path)
    for scan in scans:
        #Конкретный скан
        scan_path = os.path.join(run_data_path, scan)
        
        #Проверка, что это действительно файл фМРТ
        if scan.find('.nii') != -1:
            scan_img = nib.load(scan_path)
            scans_list.append(scan_img.get_fdata())
    return scans_list

def load_patient_data(patient_data_path, mode='+', ban_names=['log', 'structural']):
    patient_data_dict = {}

    os.chdir(patient_data_path)
    
    #Прогоны для одного пациента
    runs = os.listdir(path=patient_data_path)
    for run in runs:
        #Конкретный прогон
        run_path = os.path.join(patient_data_path, run)
        if run in ban_names:
            continue
        #Проверка, что это одна из нужных папок папка 
        if run.find('.') == -1 and len(run.split('_')) > 1 and run.split('_')[1].isnumeric():
            if mode == 'fr' and run.split('_')[0] == 'str':
                continue
            if mode == 'str' and run.split('_')[0] == 'fr':
                continue
            
            #Список сканов
            scans = os.listdir(path=run_path)
            for scan in scans:
                
                #Конкретный скан
                scan_path = os.path.join(run_path, scan)
                
                #Проверка, что это действительно файл фМРТ
                if scan.find('.nii') != -1:
                    if run not in patient_data_dict.keys():
                        patient_data_dict[run] = []
                    scan_img = nib.load(scan_path)
                    patient_data_dict[run].append(scan_img.get_fdata())
    return patient_data_dict


def load_logs(logs_link, numb_patients, n_runs=3):
    #Возвращаемый словарь, ключи: пациент, прогон -> логи
    logs_dict = {}
    os.chdir(logs_link)
    list_dir = os.listdir(path=logs_link)
    i = 0
    for directory in list_dir:
        if directory == 'M_y2_002':
            continue
        if i >= numb_patients:
            break
        
        #Пациент
        patient = directory.split('_')[2:]
        if len(patient) == 1:
            patient = patient[0]
        else:
            patient = patient[0] + '_' + patient[1]
        patient_path = os.path.join(logs_link, directory)
        os.chdir(patient_path)
        
        #Прогоны для одного пациента
        files = os.listdir(path=patient_path)
        for file in files:
            
            #Конкретный прогон
            file_path = os.path.join(patient_path, file)
            if patient not in logs_dict.keys():
                logs_dict[patient] = {}
            log_file = pd.read_csv(file_path, sep='\t')
            logs_dict[patient][file] = log_file
        i = i + 1
    return logs_dict


def load_run_logs(run_logs_link):
    log_file = pd.read_csv(run_logs_link, sep='\t')
    return log_file


def load_patient_logs(patient_logs_path,):
    patient_logs_dict = {}
    os.chdir(patient_logs_path)
    
    #Прогоны для одного пациента
    files = os.listdir(path=patient_logs_path)
    for file in files:
        #Несколько прогонов сразу, нужно разбить
        file_path = os.path.join(patient_logs_path, file)
        log_file = pd.read_csv(file_path, sep='\t')
        block_nums = log_file['values.TestBlockNum'].unique()
        block_nums = block_nums[block_nums != 0]
        for block_num in block_nums:
            run_log = log_file[log_file['values.TestBlockNum'] == block_num]
            run_name = file.split('_')[-1].split('.')[0] + '_' + str(block_num)
            run_log = run_log.reset_index(drop=True)
            patient_logs_dict[run_name] = run_log
    return patient_logs_dict



def get_FRAME_TIMES(logs_file, nX=None, nY=None, nZ=None, nR=None, nT=None, nS=None):#пациент,воксель,прогон
    answer = [i*2.5 for i in range(0, nT)]
    answer = np.array(answer)
    return answer


def get_SLICE_TIMES(logs_file, nX=None, nY=None, nZ=None, nR=None, nT=None, nS=None):#пациент,воксель,прогон
    new_list = [0.1 * i for i in range(1, nZ+1)]
    new_list2 = [0 for i in range(nZ)]
    i = 0
    for j in range(nZ):
        if j % 2 == 0:
            new_list2[j] = new_list[i]
            i = i + 1
    for j in range(nZ):
        if j % 2 == 1:
            new_list2[j] = new_list[i]
            i = i + 1
    answer = copy.deepcopy(new_list2)
    answer = np.array(answer)
    return answer


def get_EVENTS(logs_file, nX=None, nY=None, nZ=None, nR=None, nT=None, nS=None):#пациент,воксель,прогон
    trial_df = logs_file
    id_list = list(trial_df['trialNum'])
    for i, _ in enumerate(id_list):
        id_list[i] += 1
    time_list = list(trial_df['offset'])
    duration_list = [2.5 for id_ex in id_list]
    height_list = [1 for id_ex in id_list]
    answer = [[id_list[i], time_list[i], duration_list[i], height_list[i]] for i in range(len(id_list))]
    answer = np.array(answer)
    return answer


def conv2d(A, B):
    A = A.T
    n1, n2 = A.shape
    m1, m2 = B.shape
    if n1-m1+1 > 0 and n2-m2+1 > 0:
        C = np.zeros((n1-m1+1, n2-m2+1))
    else: 
        D = copy.deepcopy(A)
        A = copy.deepcopy(B)
        B = copy.deepcopy(D)
        n1, n2 = A.shape
        m1, m2 = B.shape
        print(n1, n2)
        print(m1, m2)
        C = np.zeros((n1-m1+1, n2-m2+1))
    for i in range(n1-m1+1):
        for j in range(n2-m2+1):
            summ = 0
            stroka = f'C[{i},{j}]='
            for u in range(m1):
                for v in range(m2):
                    #print(u,v)
                    summ = summ + A[i+u,j+v] * B[u,v]
                    stroka = stroka + f'{A[i+u,j+v]}*{B[u,v]}+'
            stroka = stroka[:-1]
            stroka = stroka + f'={summ}'
            #print(stroka)
            C[i,j] = summ
    return C


def h1(t, alpha1=6, alpha2=16, beta1=1, beta2=1, c=1/6, A=1):
    gamma1 = t**(alpha1 - 1) * beta1**alpha1 * math.exp(-beta1 * t) / math.gamma(alpha1)
    gamma2 = t**(alpha2 - 1) * beta2**alpha2 * math.exp(-beta2 * t) / math.gamma(alpha2)
    return A * (gamma1 - gamma2 * c)




def h2(t, a1=6, a2=12, b1=0.9, b2=0.9, c=0.35):
    d1 = a1 * b1
    d2 = a2 * b2
    gamma1 = (t/d1)**a1 * np.exp(-(t-d1)/b1)
    gamma2 = (t/d2)**a2 * np.exp(-(t-d2)/b2)
    return gamma1 - c * gamma2



def s(t, t_min, t_max):
    if t >= t_min and t <= t_max:
        return 1
    else:
        return 0



def x_coef1(t, t_min=0, t_max=0, alpha1=6, alpha2=16, beta1=1, beta2=1, c=1/6, A=1):
    #return integrate.quad(lambda x: h(x, alpha1=alpha1, alpha2=alpha2, beta1=beta1, beta2=beta2, c=c, A=A) * s(t-x, t_min, t_max), 0, 10)[0]
    return integral(lambda x: h1(x, alpha1=alpha1, alpha2=alpha2, beta1=beta1, beta2=beta2, c=c, A=A) * \
                    s(t-x, t_min, t_max), 0, t, step = 0.01)


def x_coef2(t, t_min=0, t_max=0, alpha1=6, alpha2=16, beta1=1, beta2=1, c=1/6, A=1):
    #return integrate.quad(lambda x: h(x, alpha1=alpha1, alpha2=alpha2, beta1=beta1, beta2=beta2, c=c, A=A) * s(t-x, t_min, t_max), 0, 10)[0]
    return integral(lambda x: h2(x, a1=6, a2=12, b1=0.9, b2=0.9, c=0.35) * \
                    s(t-x, t_min, t_max), 0, t, step = 0.01)




def x_coef3(t, t_min=0, t_max=0, alpha1=6, alpha2=16, beta1=1, beta2=1, c=1/6, A=1):
    #return integrate.quad(lambda x: h(x, alpha1=alpha1, alpha2=alpha2, beta1=beta1, beta2=beta2, c=c, A=A) * s(t-x, t_min, t_max), 0, 10)[0]
    return integrate.quad(lambda x: h2(x, a1=6, a2=12, b1=0.9, b2=0.9, c=0.35) * \
                          s(t-x, t_min, t_max), 0, t, limit=100, epsrel=1e-3)[0]



def x_coef4(t, t_min=0, t_max=0, alpha1=6, alpha2=16, beta1=1, beta2=1, c=1/6, A=1):
    #return integrate.quad(lambda x: h(x, alpha1=alpha1, alpha2=alpha2, beta1=beta1, beta2=beta2, c=c, A=A) * s(t-x, t_min, t_max), 0, 10)[0]
    return integrate.quad(lambda x: h1(x, alpha1=alpha1, alpha2=alpha2, beta1=beta1, beta2=beta2, c=c, A=A) * \
                          s(t-x, t_min, t_max), 0, t, limit=100, epsrel=1e-3)[0]



def integral(f, x_min, x_max, step=None):
    if step is None:
        step = (x_max - x_min) / 100
    answer = 0
    x = x_min
    while x <= x_max:
        answer = answer + f(x) * step
        x = x + step
    return answer



def design_matrix(frametimes, slicetimes=0, events=np.array([[1, 0]]), S=None, 
               hrf_parameters=np.array([[5.4, 5.2, 10.8, 7.35, 0.35]]), shift=None, voxel=None,
                nX=None, nY=None, nZ=None, nR=None, nT=None, nS=None):
    z = voxel[2]
    nd = nT
    matrix = np.zeros((nT, nS))
    eventime = events[1]
    evendur = events[2]
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if eventime[j] != -1:
                t = frametimes[i] + slicetimes[z]
                #print(eventime[j])
                #print(evendur[j])
                #print(matrix[i, j])
                matrix[i, j] = x_coef4(t, t_min=eventime[j], t_max=eventime[j]+evendur[j])
    return matrix




def print_matrix(matrix):
    m, n = matrix.shape
    for i in range(m):
        for j in range(n):
            elem = matrix[i, j]
            print("{0:0.2f}".format(elem), end=', ')
        print()




def get_design_matrix(logs_file, voxel, nX=None, nY=None, nZ=None, nR=None, nT=None, nS=None, frametimes=None, slicetimes=None, events=None):
    if frametimes is None:
        FRAME_TIMES = get_FRAME_TIMES(logs_file, nX, nY, nZ, nR, nT, nS)
    else:
        FRAME_TIMES = frametimes
    if slicetimes is None:
        SLICE_TIMES = get_SLICE_TIMES(logs_file, nX, nY, nZ, nR, nT, nS)
    else:
        SLICE_TIMES = slicetimes
    if events is None:
        EVENTS = get_EVENTS(logs_file, nX, nY, nZ, nR, nT, nS)
    else:
        EVENTS = events
    #print('getting design matrix')
    #print("FRAME_TIMES:\n", FRAME_TIMES)
    #print("SLICE_TIMES:\n", SLICE_TIMES)
    #print("EVENTS:\n", EVENTS)
    #answer_dict = fmridesign2(FRAME_TIMES, SLICE_TIMES, EVENTS)
    #DESIGN_MATRIX = answer_dict['X'][:, :, 0]
    DESIGN_MATRIX = design_matrix(FRAME_TIMES, SLICE_TIMES, EVENTS, voxel=voxel, \
                                  nX=nX, nY=nY, nZ=nZ, nR=nR, nT=nT, nS=nS)
    #print('Размеры:', DESIGN_MATRIX.shape)
    return DESIGN_MATRIX



def pre_whitening(array_matrixes, rho=0.5):
    new_array_matrixes = np.zeros(array_matrixes.shape)
    for i in range(len(array_matrixes)):
        if i == 0:
            new_array_matrixes[i] = array_matrixes[i]
        else:
            new_array_matrixes[i] = (array_matrixes[i] - rho * array_matrixes[i-1])/(1 - rho**2)**0.5
    return new_array_matrixes



def get_X(patient_logs_path=None, voxel=None, nX=None, nY=None, nZ=None, 
          nR=None, nT=None, nS=None, runs=None, frametimes=None, slicetimes=None, events=None, prew=True):
    #print('Получаем матрицы X')
    logs_dict = load_patient_logs(patient_logs_path)
    rho = 0.5
    X_list = []
    for run in runs:
        #display(logs_files[r])
        t1 = time.time()
        X = get_design_matrix(logs_dict[run], voxel, nX, nY, nZ, nR, nT, nS, \
                              frametimes=frametimes[run], slicetimes=slicetimes[run], events=events[run])
        t2 = time.time()
        #print('Одна матрица X:', t2-t1)
        #print(X)
        X_list.append(copy.deepcopy(X))
        t3 = time.time()
        #print('С копированием:', t3-t1)
        #print
    X_array = np.array(X_list)
    if prew:
        #print('До предобработки')
        #print(X_array)
        X_prew_array = pre_whitening(X_array, rho)
        #print('После предобработки')
        #print(X_prew_array)
        return X_prew_array
    else:
        return X_array




def get_Y(patient_data_path=None, voxel=None, Y_paths=None, runs=None, prew=True):
    rho = 0.5
    Y_list = []
    t1 = time.time()
    if Y_paths == None:
        patient_data_dict = load_patient_data(patient_data_path)
        #print(patient_data_dict.keys())
        for run in runs:
            Y_list.append(np.array([scan[voxel] for scan in patient_data_dict[run]]))
    else:
        for Y_path in Y_paths:
            Y_list.append(np.load(Y_path))
    t2 = time.time()
    #print('Y Загружен за', t2-t1)
    Y_array = np.array(Y_list)
    if prew:
        Y_prew_array = pre_whitening(Y_array, rho)
        t3 = time.time()
        #print('Обработан за', t3-t1)
        return Y_prew_array
    else:
        t3 = time.time()
        return Y_array



def MSE(v1, v2):
    if v1.shape != v2.shape:
        print("Векторы разной длины: ", v1.shape, " != ", v2.shape)
        return None
    answer = 0
    for i in range(len(v1)):
        diff = (v1[i] - v2[i])**2
        answer = answer + diff
    return answer


def avg_diff2(matrix1, matrix2):
    if matrix1.shape != matrix2.shape:
        print('Разный размер')
        return None
    summ = 0
    for i in range(matrix1.shape[0]):
        for j in range(matrix1.shape[1]):
            summ += abs(matrix1[i][j] - matrix2[i][j])
    #print(summ, matrix1.shape)
    return summ / (matrix1.shape[0] * matrix1.shape[1])



def avg_diff3(matrix1, matrix2):
    if matrix1.shape != matrix2.shape:
        print('Разный размер')
        return None
    summ = 0
    for i in range(matrix1.shape[0]):
        for j in range(matrix1.shape[1]):
            for k in range(matrix1.shape[2]):
                summ += abs(matrix1[i][j][k] - matrix2[i][j][k])
    #print(summ, matrix1.shape)
    return summ / (matrix1.shape[0] * matrix1.shape[1] * matrix1.shape[2])

def EM_GLM(X_list, y_list, x_star, nX=None, nY=None, nZ=None, nR=None, nT=None, nS=None, max_iter=100, tol=1e-6):
    """
    Параметры:
    - X_list: список матриц дизайна [X_1, ..., X_M]
    - y_list: список данных [y_1, ..., y_M]
    - x_star_list: список ковариат [x_1^*, ..., x_M^*]
    """
    M = len(X_list)
    p = X_list[0].shape[1]  # Размерность beta_i
    N = X_list[0].shape[0]  # Число наблюдений в запуске
    
    # Инициализация
    B = np.zeros((p, 1))  # B.shape = (p, k), где k — размерность x_i^*
    Omega = np.eye(p)
    sigma_sq = [1.0] * M
    
    for _ in range(max_iter):
        # E-шаг: вычисление апостериорных средних и ковариаций
        beta_hat = []
        H = []
        for i in range(M):
            X, y = X_list[i], y_list[i]
            mu_i = B.T @ x_star
            Sigma_inv = np.linalg.inv(Omega) + (1 / sigma_sq[i]) * X.T @ X
            Sigma = np.linalg.inv(Sigma_inv)
            beta_hat_i = Sigma @ (np.lianlg.inv(Omega) @ mu_i + (1 / sigma_sq[i]) * X.T @ y)
            beta_hat.append(beta_hat_i)
            H.append(Sigma)
        
        # M-шаг: обновление параметров
        B_new = np.linalg.inv(sum(np.outer(x_star, x_star) for i in range(M))) @ \
                sum(x_star @ beta_hat[i].T for i in range(M))
        
        Omega_new = (1 / M) * sum(
            H[i] + np.outer(beta_hat[i] - B_new.T @ x_star, 
                            beta_hat[i] - B_new.T @ x_star) 
            for i in range(M)
        )
        
        sigma_sq_new = [
            (1 / N) * (
                np.linalg.norm(y_list[i] - X_list[i] @ beta_hat[i])**2 +
                np.trace(X_list[i] @ H[i] @ X_list[i].T)
            )
            for i in range(M)
        ]
        
        # Проверка сходимости
        if (np.linalg.norm(B_new - B) < tol) and \
            np.linalg.norm(Omega_new - Omega) < tol and \
            np.all(np.abs(np.array(sigma_sq_new) - np.array(sigma_sq)) < tol):
            break
        
        B, Omega, sigma_sq = B_new, Omega_new, sigma_sq_new
    
    return beta_hat, sigma_sq


def EM_GLM_enhanced(X_list, y_list, x_star, nX, nY, nZ, nR, nT, nS, max_iter=100, tol=1e-6, alpha_B=0.1, alpha_Omega=1.0):
    """
    Улучшенный EM-алгоритм для GLM с регуляризацией, но без параллелизма.

    Параметры:
    - X_list: список матриц дизайна (разреженные или плотные)
    - y_list: список векторов ответов
    - x_star_list: список ковариат для межгрупповых эффектов
    - max_iter: максимальное число итераций
    - tol: критерий сходимости
    - alpha_B: параметр регуляризации для B
    - alpha_Omega: параметр регуляризации для Omega
    """
    M = len(X_list)
    p = X_list[0].shape[1]
    k = x_star.shape[0]  # Размерность x_i^*
    N = X_list[0].shape[0]

    # Инициализация параметров
    B = np.zeros((p, k))
    Omega = np.eye(p)
    sigma_sq = np.ones(M)

    for _ in range(max_iter):
        # E-шаг: вычисление апостериорных параметров
        beta_hat = []
        H = []
        for i in range(M):
            X = X_list[i]
            if issparse(X):
                XTX = X.T @ X
            else:
                XTX = X.T @ X

            Sigma_inv = inv(Omega) + (1 / sigma_sq[i]) * XTX
            Sigma = inv(Sigma_inv)
            mu_i = B.T @ x_star[i, :]
            beta_hat_i = Sigma @ (inv(Omega) @ mu_i + (1 / sigma_sq[i]) * X.T @ y_list[i])
            beta_hat.append(beta_hat_i)
            H.append(Sigma)

        # M-шаг с регуляризацией
        # Обновление B с L2-регуляризацией
        sum_xx = np.zeros((k, k))
        sum_xbeta = np.zeros((k, p))
        for i in range(M):
            sum_xx += x_star.T @ x_star
            sum_xbeta += x_star @ beta_hat[i]
        B_new = solve(sum_xx + alpha_B * np.eye(k), sum_xbeta).T

        # Обновление Omega с априорным распределением
        sum_terms = np.zeros((p, p))
        for i in range(M):
            residual = beta_hat[i] - B_new.T @ x_star[i, :]
            sum_terms += H[i] + np.outer(residual, residual)
        Omega_new = (sum_terms + alpha_Omega * np.eye(p)) / (M + alpha_Omega + p + 1)

        # Обновление sigma_sq
        sigma_sq_new = []
        for i in range(M):
            X = X_list[i]
            residual = y_list[i] - X @ beta_hat[i]
            sigma_sq_new.append(
                (np.dot(residual, residual) + np.trace(X @ H[i] @ X.T)) / N
            )

        # Проверка сходимости
        if (np.linalg.norm(B_new - B) < tol and
            np.linalg.norm(Omega_new - Omega) < tol and
            np.all(np.abs(np.array(sigma_sq_new) - sigma_sq) < tol)):
            break

        B, Omega, sigma_sq = B_new, Omega_new, np.array(sigma_sq_new)

    return beta_hat, sigma_sq

def em_algorithm_glm(X_list, y_list, X_star, nX, nY, nZ, nR, nT, nS, max_iter=100, tol=1e-6, 
                    alpha_ridge=0.1, sparse=False):
    """
    Реализация EM-алгоритма для оценки параметров регрессии GLM 
    с регуляризацией и поддержкой разреженных матриц.

    Параметры:
    - X_list: список матриц дизайна для каждого прогона (run).
    - y_list: список векторов откликов для каждого прогона.
    - X_star: матрица дизайна для эффектов задач (tasks).
    - max_iter: максимальное число итераций.
    - tol: критерий сходимости.
    - alpha_ridge: параметр регуляризации L2 (Ridge).
    - sparse: использовать разреженные матрицы.

    Возвращает:
    - B: оценки параметров для эффектов задач.
    - Omega: ковариационная матрица параметров.
    - sigma_sq_list: список оценок дисперсий ошибок для каждого прогона.
    """
    M = len(X_list)  # Количество прогонов
    if M == 0:
        raise ValueError("X_list не может быть пустым")
    
    k = X_list[0].shape[1]  # Количество параметров в beta_i
    k_star = X_star.shape[1]  # Количество параметров в B
    n_list = [X.shape[0] for X in X_list]  # Число наблюдений в каждом прогоне

    # Проверка согласованности размеров
    if X_star.shape[0] != M:
        raise ValueError(f"X_star должен иметь {M} строк (по одной на каждый прогон), но имеет {X_star.shape[0]}")

    # Преобразование в разреженные матрицы при необходимости
    if sparse:
        X_list = [csr_matrix(X) if not issparse(X) else X for X in X_list]
        X_star = csr_matrix(X_star) if not issparse(X_star) else X_star

    # Инициализация параметров
    B = np.zeros((k_star, k))
    Omega = np.eye(k)
    sigma_sq_list = [1.0 for _ in range(M)]
    mu_list = [np.zeros(k) for _ in range(M)]

    # Функция для решения линейной системы с учетом разреженности
    def solve_linear_system(A, b):
        if sparse and issparse(A):
            return spsolve(A, b)
        else:
            return np.linalg.solve(A, b)

    for iteration in range(max_iter):
        # E-шаг: вычисление апостериорных средних и ковариаций
        h_list = []
        H_list = []
        for i in range(M):
            X_i = X_list[i]
            y_i = y_list[i]
            sigma_sq_i = sigma_sq_list[i]
            mu_i = mu_list[i]

            # Вычисление апостериорного среднего и ковариации
            inv_Omega = inv(Omega) if not sparse else csr_matrix(inv(Omega))
            
            # Добавление регуляризации Ridge
            reg_term = alpha_ridge * np.eye(k)
            if sparse:
                reg_term = csr_matrix(reg_term)
            
            term = inv_Omega + (1 / sigma_sq_i) * X_i.T @ X_i + reg_term
            
            # Решение линейной системы
            if sparse and issparse(term):
                term = term.tocsc()  # Преобразование для эффективного решения
            H_i = solve_linear_system(term, np.eye(k))
            
            right_part = inv_Omega @ mu_i + (1 / sigma_sq_i) * X_i.T @ y_i
            h_i = H_i @ right_part

            h_list.append(h_i)
            H_list.append(H_i)

        # M-шаг: обновление параметров
        # Обновление B с регуляризацией Ridge
        Z = np.vstack(h_list)  # Матрица апостериорных средних (M x k)
        
        # Проверка размерностей
        if X_star.shape[0] != Z.shape[0]:
            raise ValueError(f"Несоответствие размеров: X_star имеет {X_star.shape[0]} строк, а Z имеет {Z.shape[0]} строк")
        
        if alpha_ridge > 0:
            # Используем Ridge регрессию отдельно для каждого параметра
            B_new = np.zeros((k_star, k))
            for j in range(k):
                ridge = Ridge(alpha=alpha_ridge, fit_intercept=False)
                ridge.fit(X_star, Z[:, j])
                B_new[:, j] = ridge.coef_
        else:
            # Стандартное решение без регуляризации
            if sparse and issparse(X_star):
                B_new = np.zeros((k_star, k))
                for j in range(k):
                    B_new[:, j] = spsolve(X_star.T @ X_star, X_star.T @ Z[:, j])
            else:
                B_new = np.linalg.solve(X_star.T @ X_star, X_star.T @ Z)

        # Обновление mu_i
        mu_list_new = [B_new.T @ X_star[i, :] for i in range(M)]

        # Обновление Omega с добавлением небольшой диагональной нагрузки для устойчивости
        sum_H_hh = sum([H_list[i] + np.outer(h_list[i], h_list[i]) for i in range(M)])
        mu_sum = sum(mu_list_new)
        Omega_new = (1 / (M - k_star)) * (sum_H_hh - (1 / M) * np.outer(mu_sum, mu_sum))
        
        # Добавление небольшой диагональной нагрузки
        Omega_new += 1e-6 * np.eye(k)

        # Обновление sigma_sq_i
        sigma_sq_list_new = []
        for i in range(M):
            X_i = X_list[i]
            y_i = y_list[i]
            beta_i = h_list[i]
            residual = y_i - X_i @ beta_i
            sigma_sq_i_new = np.sum(residual ** 2) / (n_list[i] - k)
            sigma_sq_list_new.append(sigma_sq_i_new)

        # Проверка сходимости
        delta_B = np.max(np.abs(B_new - B))
        delta_Omega = np.max(np.abs(Omega_new - Omega))
        delta_sigma = np.max(np.abs(np.array(sigma_sq_list_new) - np.array(sigma_sq_list)))

        if delta_B < tol and delta_Omega < tol and delta_sigma < tol:
            print(f"Сходимость достигнута на итерации {iteration + 1}.")
            break

        # Обновление параметров
        B = B_new
        Omega = Omega_new
        sigma_sq_list = sigma_sq_list_new
        mu_list = mu_list_new

    return h_list[-1], H_list[-1]

def em_glm(X_list, y_list, X_star, nX, nY, nZ, nR, nT, nS, max_iter=100, tol=1e-6, alpha_ridge=0.1, sparse=False):
    """
    EM-алгоритм для GLM с регуляризацией и поддержкой разреженных матриц.

    Параметры:
    - X_list: список матриц дизайна (каждая n_i × k)
    - y_list: список векторов откликов (каждый длины n_i)
    - X_star: матрица задач (M × k^*)
    - alpha_ridge: параметр L2-регуляризации
    - sparse: использовать разреженные матрицы

    Возвращает:
    - B: оценки параметров задач (k^* × k)
    - Omega: ковариационная матрица (k × k)
    - sigma_sq: список дисперсий ошибок
    """
    M = len(X_list)
    k = X_list[0].shape[1]
    k_star = X_star.shape[1]
    n_list = [X.shape[0] for X in X_list]

    # Инициализация
    B = np.zeros((k_star, k))
    Omega = np.eye(k)
    sigma_sq = [1.0] * M
    mu_list = [np.zeros(k) for _ in range(M)]

    # Преобразование в разреженные матрицы
    if sparse:
        X_list = [csr_matrix(X) if not issparse(X) else X for X in X_list]
        X_star = csr_matrix(X_star) if not issparse(X_star) else X_star

    for iteration in range(max_iter):
        # E-шаг
        h_list, H_list = [], []
        for i in range(M):
            X_i, y_i = X_list[i], y_list[i]
            inv_Omega = inv(Omega) if not sparse else csr_matrix(inv(Omega))
            
            # Регуляризованная система
            A = inv_Omega + (1/sigma_sq[i]) * X_i.T @ X_i + alpha_ridge * np.eye(k)
            b = inv_Omega @ mu_list[i] + (1/sigma_sq[i]) * X_i.T @ y_i
            
            if sparse:
                A = A.tocsc()
                h_i = spsolve(A, b)
                H_i = spsolve(A, np.eye(k))
            else:
                h_i = np.linalg.solve(A, b)
                H_i = np.linalg.solve(A, np.eye(k))
            
            h_list.append(h_i)
            H_list.append(H_i)

        # M-шаг
        Z = np.vstack(h_list)
        
        # Обновление B с Ridge-регуляризацией
        if alpha_ridge > 0:
            ridge = Ridge(alpha=alpha_ridge, fit_intercept=False)
            ridge.fit(X_star, Z)
            B_new = ridge.coef_.T
        else:
            if sparse:
                B_new = spsolve(X_star.T @ X_star, X_star.T @ Z)
            else:
                B_new = np.linalg.solve(X_star.T @ X_star, X_star.T @ Z)

        # Обновление Omega с диагональной нагрузкой
        mu_list_new = [B_new.T @ X_star[i] for i in range(M)]
        mu_sum = sum(mu_list_new)
        Omega_new = (1/(M - k_star)) * ( \
            sum(H_list[i] + np.outer(h_list[i], h_list[i]) for i in range(M)) - \
            (1/M) * np.outer(mu_sum, mu_sum))
        Omega_new += 1e-6 * np.eye(k)  # Для устойчивости

        # Обновление sigma_sq
        sigma_sq_new = [
            np.sum((y_list[i] - X_list[i] @ h_list[i])**2) / (n_list[i] - k)
            for i in range(M)
        ]

        # Проверка сходимости
        if all([
            np.max(np.abs(B_new - B)) < tol,
            np.max(np.abs(Omega_new - Omega)) < tol,
            np.max(np.abs(np.array(sigma_sq_new) - np.array(sigma_sq))) < tol
        ]):
            print(f"Сходимость на итерации {iteration + 1}")
            break

        B, Omega, sigma_sq = B_new, Omega_new, sigma_sq_new
        mu_list = mu_list_new

    return h_list[-1], H_list[-1]

def accelerated_em_algorithm(X_list, Y_list, X_star, nX, nY, nZ, nR, nS, nT, max_iter=500, tol=1e-8,
                           min_sigma2=1e-6, ridge_alpha=1e-4, adaptative_step=True):
    """
    Ускоренная версия EM-алгоритма с адаптивным шагом и улучшенной сходимостью
    
    Параметры:
    - adaptative_step: использовать адаптивный шаг для ускорения сходимости
    - min_sigma2: минимальное значение дисперсии
    - ridge_alpha: параметр регуляризации
    - tol: более строгий критерий сходимости
    """
    
    # 1. Улучшенная инициализация
    h = np.zeros((nR, nS))
    H = np.array([np.eye(nS)*0.1 for _ in range(nR)])
    
    # Инициализация параметров с учетом масштаба данных
    init_var = np.mean([np.var(y) for y in Y_list])
    Omega = np.eye(nS) * init_var
    Sigma2 = np.array([max(np.var(y), min_sigma2) for y in Y_list])
    Mu = np.zeros((nR, nS))
    
    # Для адаптивного шага
    prev_h = h.copy()
    prev_H = H.copy()
    step_size = 1.0
    momentum = 0.5  # Параметр инерции
    
    for n_iter in range(max_iter):
        # Сохраняем предыдущие значения для расчета изменений
        old_h = h.copy()
        old_H = H.copy()
        
        # E-шаг с адаптивным обучением
        for i in range(nR):
            Xi = X_list[i]
            yi = Y_list[i]
            sigma2 = max(Sigma2[i], min_sigma2)
            
            try:
                # Регуляризованное вычисление
                inv_term = pinv(Omega) + (1/sigma2) * (Xi.T @ Xi) + ridge_alpha * np.eye(nS)
                H_i = pinv(inv_term)
                
                # Адаптивный расчет h
                h_i = H_i @ (pinv(Omega) @ Mu[i] + (1/sigma2) * Xi.T @ yi)
                
                if adaptative_step:
                    # Применяем адаптивный шаг с инерцией
                    h[i] = h_i * step_size + prev_h[i] * (1 - momentum)
                    H[i] = H_i * step_size + prev_H[i] * (1 - momentum)
                else:
                    h[i] = h_i
                    H[i] = H_i
                    
            except:
                h[i] = old_h[i]
                H[i] = old_H[i]
        
        # M-шаг с улучшенной устойчивостью
        try:
            # Регуляризованная оценка B
            ridge = Ridge(alpha=ridge_alpha, fit_intercept=False)
            ridge.fit(X_star, h)
            B = ridge.coef_.T
            
            # Обновление Mu
            Mu = (X_star @ B)
            
            # Обновление Omega с диагональной нагрузкой
            Mu_mean = np.mean(Mu, axis=0)
            Omega_numerator = sum(
                [H[i] + np.outer(h[i] - Mu_mean, h[i] - Mu_mean) 
                 for i in range(nR)]
            )
            Omega = Omega_numerator / nR + ridge_alpha * np.eye(nS)
            
            # Обновление Sigma2 с защитой
            for i in range(nR):
                resid = Y_list[i] - X_list[i] @ h[i]
                Sigma2[i] = max(np.mean(resid**2), min_sigma2)
                
        except:
            # Откат при ошибках
            h = old_h.copy()
            H = old_H.copy()
        
        # Адаптивная регулировка шага
        if adaptative_step:
            h_diff = np.max(np.abs(h - old_h))
            if h_diff < 1e-4:
                step_size = min(1.5 * step_size, 2.0)  # Увеличиваем шаг
            elif h_diff > 0.1:
                step_size = max(0.5 * step_size, 0.1)  # Уменьшаем шаг
        
        # Проверка сходимости
        h_diff = np.max(np.abs(h - old_h))
        H_diff = np.max([np.max(np.abs(H[i] - old_H[i])) for i in range(nR)])
        
        if n_iter % 10 == 0:
            print(f"Iter {n_iter}: h_diff={h_diff:.2e}, H_diff={H_diff:.2e}, step={step_size:.2f}")
        
        if h_diff < tol and H_diff < tol and n_iter > 5:
            print(f"Сходимость достигнута на итерации {n_iter}")
            print(f"Iter {n_iter}: h_diff={h_diff:.2e}, H_diff={H_diff:.2e}, step={step_size:.2f}")
            print
            break
            
        prev_h = old_h.copy()
        prev_H = old_H.copy()
    
    return h, H


def robust_em_algorithm(X_list, Y_list, X_star, nX, nY, nZ, nR, nS, nT, max_iter=500, tol=1e-8,
                       min_sigma2=1e-6, init_ridge=1e-4, noise_scale=1e-3):
    """
    Устойчивый EM-алгоритм с защитой от застревания
    
    Параметры:
    - noise_scale: величина шума для выхода из локальных оптимумов
    - init_ridge: начальное значение регуляризации
    """
    
    # 1. Инициализация с добавлением небольшого шума
    h = np.random.normal(0, 0.1, size=(nR, nS))
    H = np.array([np.eye(nS)*(0.1 + np.random.rand()*0.01) for _ in range(nR)])
    
    # Динамическая регуляризация
    ridge_alpha = init_ridge
    Omega = np.eye(nS) * np.mean([np.var(y) for y in Y_list])
    Sigma2 = np.array([max(np.var(y), min_sigma2) for y in Y_list])
    Mu = np.random.normal(0, 0.1, size=(nR, nS))
    
    for n_iter in range(max_iter):
        old_h = h.copy()
        old_H = H.copy()
        
        # E-шаг с адаптивной регуляризацией
        for i in range(nR):
            Xi = X_list[i]
            yi = Y_list[i]
            
            # Динамическое уменьшение регуляризации
            current_ridge = max(ridge_alpha * 0.9**n_iter, 1e-8)
            
            try:
                # Устойчивое обращение матрицы
                inv_term = pinv(Omega) + (1/max(Sigma2[i], min_sigma2)) * (Xi.T @ Xi) + current_ridge * np.eye(nS)
                H_i = pinv(inv_term)
                
                # Обновление с добавлением шума
                h_update = H_i @ (pinv(Omega) @ Mu[i] + (1/max(Sigma2[i], min_sigma2)) * Xi.T @ yi)
                h[i] = h_update + np.random.normal(0, noise_scale * max(1, np.abs(h_update.mean())))
                H[i] = H_i
                
            except:
                # При ошибке добавляем больше шума
                h[i] = old_h[i] + np.random.normal(0, noise_scale*2)
                H[i] = old_H[i]
        
        # M-шаг с проверкой обновлений
        try:
            # Обновление B с проверкой ранга
            XtX = X_star.T @ X_star
            if np.linalg.matrix_rank(XtX) < XtX.shape[0]:
                XtX += ridge_alpha * np.eye(XtX.shape[0])
            
            B = pinv(XtX) @ X_star.T @ h
            
            # Обновление Mu с контролем изменений
            new_Mu = X_star @ B
            if not np.allclose(new_Mu, Mu, atol=1e-6):
                Mu = new_Mu
            else:
                Mu = new_Mu + np.random.normal(0, noise_scale, size=Mu.shape)
            
            # Обновление Omega с контролем сингулярности
            Mu_mean = np.mean(Mu, axis=0)
            Omega_numerator = sum(
                [H[i] + np.outer(h[i] - Mu_mean, h[i] - Mu_mean) 
                 for i in range(nR)]
            )
            new_Omega = Omega_numerator / nR
            
            # Проверка положительной определенности
            min_eigval = np.linalg.eigvalsh(new_Omega).min()
            if min_eigval > 1e-8:
                Omega = new_Omega
            else:
                Omega = new_Omega + (1e-6 - min_eigval) * np.eye(nS)
            
            # Обновление Sigma2 с защитой
            for i in range(nR):
                resid = Y_list[i] - X_list[i] @ h[i]
                new_sigma = max(np.mean(resid**2), min_sigma2)
                if new_sigma != Sigma2[i]:
                    Sigma2[i] = new_sigma
                else:
                    Sigma2[i] = new_sigma * (1 + np.random.rand()*0.01)
                    
        except Exception as e:
            print(f"Ошибка на M-шаге: {str(e)}")
            # Применяем более агрессивное возмущение
            h = old_h + np.random.normal(0, noise_scale*3, size=h.shape)
            H = old_H
            ridge_alpha *= 2  # Увеличиваем регуляризацию
        
        # Проверка изменений
        h_diff = np.max(np.abs(h - old_h))
        H_diff = np.max([np.max(np.abs(H[i] - old_H[i])) for i in range(nR)])
        
        print(f"Iter {n_iter}: h_diff={h_diff:.2e}, H_diff={H_diff:.2e}, ridge={current_ridge:.1e}")
        
        # Динамическое уменьшение шума
        noise_scale = max(noise_scale * 0.99, 1e-8)
        
        if h_diff < tol and H_diff < tol:
            print(f"Сходимость достигнута на итерации {n_iter}")
            break
    
    return h, H

def run_EM_improved(X_list, Y_list, X_star, nX=None, nY=None, nZ=None, nR=None, nT=None, nS=None, 
                   max_iter=200, tol=1e-6, min_sigma2=1e-6, ridge_alpha=1e-6):
    """
    Улучшенная версия EM-алгоритма с защитой от NaN значений
    
    Параметры:
    - min_sigma2: минимальное значение для Sigma2 (защита от деления на 0)
    - ridge_alpha: параметр регуляризации для устойчивости матричных операций
    """
    
    t1 = time.time()
    
    # Инициализация параметров с защитой от вырожденности
    h = np.zeros((nR, nS))  # Более стабильная инициализация нулями
    H = np.zeros((nR, nS, nS))
    for r in range(nR):
        H[r] = np.eye(nS) * 0.1  # Диагональная инициализация
    
    k_star = X_star.shape[1]
    M = nR
    k = nS
    
    # Инициализация с защитой
    Omega = np.eye(nS) * 0.1
    Sigma2 = np.array([max(np.var(y), min_sigma2) for y in Y_list])  # Защита от нулевой дисперсии
    Mu = np.zeros((nR, nS))  # Инициализация нулями более стабильна
    
    curr_vars = {'Omega': Omega, 'Sigma2': Sigma2, 'Mu': Mu, 'h': h, 'H': H}
    prev_vars = copy.deepcopy(curr_vars)
    
    n_iter = 0
    converged = False
    
    while not converged and n_iter < max_iter:
        prev_vars = copy.deepcopy(curr_vars)
        
        # E-шаг с защитой от вырожденности
        for i in range(nR):
            Xi = X_list[i]
            yi = Y_list[i]
            Mui = prev_vars['Mu'][i]
            Sigma2i = max(prev_vars['Sigma2'][i], min_sigma2)  # Защита
            
            try:
                # Добавляем регуляризацию для устойчивости
                inv_term = inv(prev_vars['Omega']) + (1/Sigma2i) * (Xi.T @ Xi) + ridge_alpha * np.eye(nS)
                H[i] = inv(inv_term)  # Используем псевдообратную матрицу
                
                h[i] = H[i] @ (inv(prev_vars['Omega']) @ Mui + (1/Sigma2i) * Xi.T @ yi)
            except:
                # В случае ошибки сохраняем предыдущие значения
                H[i] = prev_vars['H'][i]
                h[i] = prev_vars['h'][i]
        
        curr_vars['h'] = h.copy()
        curr_vars['H'] = H.copy()
        
        # M-шаг с защитой
        try:
            # Регуляризованное решение для B
            XtX = X_star.T @ X_star + ridge_alpha * np.eye(k_star)
            B = pinv(XtX) @ X_star.T @ curr_vars['h']
            
            for i in range(nR):
                curr_vars['Mu'][i] = B.T @ X_star[i, :]
                
            # Обновление Omega с защитой
            Mu_sum = sum(curr_vars['Mu'])
            Omega_numerator = sum([curr_vars['H'][i] + np.outer(curr_vars['h'][i], curr_vars['h'][i]) 
                              for i in range(nR)]) - (1/M) * np.outer(Mu_sum, Mu_sum)
            
            # Добавляем диагональную нагрузку
            curr_vars['Omega'] = (1/(M-k_star)) * Omega_numerator + ridge_alpha * np.eye(nS)
            
            # Обновление Sigma2 с защитой
            for i in range(nR):
                yi = Y_list[i]
                Xi = X_list[i]
                hi = curr_vars['h'][i]
                residual = yi - Xi @ hi
                curr_vars['Sigma2'][i] = max(np.mean(residual**2), min_sigma2)
                
        except:
            # В случае ошибки сохраняем предыдущие оценки
            curr_vars = copy.deepcopy(prev_vars)
        
        # Проверка сходимости
        h_diff = np.max(np.abs(curr_vars['h'] - prev_vars['h']))
        H_diff = np.max([np.max(np.abs(curr_vars['H'][i] - prev_vars['H'][i])) for i in range(nR)])
        sigma_diff = np.max(np.abs(curr_vars['Sigma2'] - prev_vars['Sigma2']))
        
        converged = (h_diff < tol) and (H_diff < tol) and (sigma_diff < tol)
        n_iter += 1
    
    t2 = time.time()
    print(f'Остановились на {n_iter} итерациях')
    print(f'Время выполнения: {t2-t1:.2f} секунд')
    print(f'Максимальное изменение h: {h_diff:.2e}')
    print(f'Максимальное изменение H: {H_diff:.2e}')
    print(f'Максимальное изменение Sigma2: {sigma_diff:.2e}')
    
    return curr_vars['h'], curr_vars['H']


def glm_em(X_list, Y_list, X_star, nX, nY, nZ, nR, nT, nS, max_iter=100, tol=1e-6):
    """
    EM-алгоритм для двухуровневой GLM.
    
    Параметры:
        Y_list (list of np.array): Данные fMRI для каждой сессии.
        X_list (list of np.array): Матрицы дизайна уровня 1.
        X_star (np.array): Ковариаты уровня 2 (e.g., задачи).
        max_iter (int): Максимум итераций.
        tol (float): Критерий сходимости.
    
    Возвращает:
        B (np.array): Групповые эффекты.
        Omega (np.array): Ковариация между сессиями.
        sigma_sq (list): Дисперсии шума.
        beta_hat (list): Оценки beta_i.
    """
    M = len(Y_list)  # Число сессий
    p = X_list[0].shape[1]  # Число предикторов уровня 1
    k = X_star.shape[1]     # Число предикторов уровня 2
    
    # Инициализация
    B = np.zeros((k, p))
    Omega = np.eye(p)
    sigma_sq = [1.0] * M
    beta_hat = [np.zeros(p) for _ in range(M)]
    
    for _ in range(max_iter):
        # E-шаг: оценка beta_i и C_i
        beta_hat_new, C_list = [], []
        for i in range(M):
            Xi, Yi = X_list[i], Y_list[i]
            Sigma_inv = inv(Omega) + (1 / sigma_sq[i]) * Xi.T @ Xi
            Sigma = inv(Sigma_inv)
            mu_i = Sigma @ (inv(Omega) @ (B.T @ X_star[i]) + (1 / sigma_sq[i]) * Xi.T @ Yi)
            beta_hat_new.append(mu_i)
            C_list.append(Sigma)
        
        # M-шаг: обновление B, Omega, sigma_sq
        B_new = inv(X_star.T @ X_star) @ X_star.T @ np.vstack(beta_hat_new)
        
        Omega_new = np.zeros((p, p))
        for i in range(M):
            diff = beta_hat_new[i] - B_new.T @ X_star[i]
            Omega_new += np.outer(diff, diff) + C_list[i]
        Omega_new /= M
        
        sigma_sq_new = []
        for i in range(M):
            Xi, Yi = X_list[i], Y_list[i]
            residual = Yi - Xi @ beta_hat_new[i]
            sigma_sq_i = (residual.T @ residual + np.trace(Xi @ C_list[i] @ Xi.T)) / len(Yi)
            sigma_sq_new.append(sigma_sq_i)
        
        # Проверка сходимости
        delta_B = np.linalg.norm(B_new - B)
        delta_Omega = np.linalg.norm(Omega_new - Omega)
        if delta_B < tol and delta_Omega < tol:
            break
        
        B, Omega, sigma_sq = B_new, Omega_new, sigma_sq_new
        beta_hat = beta_hat_new
        sigma_hat = C_list
    
    return beta_hat, sigma_hat

def print_matrix(matrix, name):
    """
    Печатает матрицу с ее именем и размером.
    
    Параметры:
        matrix (np.array): Матрица для печати.
        name (str): Имя матрицы.
    """
    n, m = matrix.shape
    print(f"{name} (shape: {matrix.shape}):")
    for i in range(n):
        for j in range(m):
            print(f"{round(matrix[i, j], 2)}", end=" ")
        print()
    print()

def print_vector(vector, name):
    """
    Печатает вектор с его именем и размером.
    
    Параметры:
        vector (np.array): Вектор для печати.
        name (str): Имя вектора.
    """
    n = vector.shape[0]
    print(f"{name} (shape: {vector.shape}):")
    for i in range(n):
        print(f"{round(vector[i], 2)}", end=" ")
    print()

def EM_new(X_list, Y_list, X_star, nX, nY, nZ, nR, nT, nS, max_iter=100, \
           tol=1e-6, min_sigma2=1e-8, min_omega_diag=1e-5, voxel=None):
    """
    Новый EM-алгоритм для GLM с улучшенной инициализацией и защитой от NaN значений.
    
    Параметры:
        X_list (list of np.array): Данные fMRI для каждой сессии.
        Y_list (list of np.array): Матрицы дизайна уровня 1.
        X_star (np.array): Ковариаты уровня 2 (e.g., задачи).
        max_iter (int): Максимум итераций.
        tol (float): Критерий сходимости.
    
    Возвращает:
        h (np.array): Оценки параметров регрессии.
        H (np.array): Ковариация между сессиями.
    """
    M = nR  # Число сессий
    k = nS  # Число предикторов уровня 1
    k_star = X_star.shape[1]  # Число предикторов уровня 2
    n = nT  # Число наблюдений в каждой сессии
    
    # Инициализация
    #h = np.zeros((M, k))
    h = np.random.normal(0, 1, size=(M, k))  # Более стабильная инициализация
    #for i in range(M):
    #    Xi = X_list[i]
    #    Yi = Y_list[i]
    #    h[i] = inv(Xi.T @ Xi) @ Xi.T @ Yi
    H = np.zeros((M, k, k))
    for i in range(M):
        H[i] = np.eye(k)
    
    #Omega = np.eye(k) * 0.1
    h_mean = np.mean(h, axis=0)
    #Omega = np.zeros((k, k))
    Omega = np.eye(k)
    #for i in range(M):
    #    Omega += np.outer(h[i] - h_mean, h[i] - h_mean)
    #Omega = np.mean(np.array([(h[i]-h_mean).T @ (h[i]-h_mean) for i in range(M)]), axis=0)
    #print(h.shape)
    #print(H.shape)
    #print(X_star.shape)
    #print(h_mean.shape)
    #print(Omega.shape)
    #Sigma2 = np.ones(M) * min_sigma2
    Sigma2 = 1/(nT - nS) * np.array([np.linalg.norm(Y_list[i] - X_list[i] @ h[i])**2 for i in range(M)])
    Mu = h.copy()
    Mu_sum = np.mean(h, axis=0)
    B = (X_star.T @ X_star) @ X_star.T @ h
    Mu_sum = np.zeros(k)
    
    for iteration in range(max_iter):
        # E-шаг: оценка h_i и H_i
        for i in range(M):
            Xi = X_list[i]
            yi = Y_list[i]
            Sigma2i = max(Sigma2[i], min_sigma2)
            try:
                inv_term = inv(Omega) + (1/Sigma2i) * (Xi.T @ Xi)
            except np.linalg.LinAlgError:
                # Если матрица вырождена, добавляем небольшую диагональную нагрузку
                print(os.getpid(), iteration, "Matrix is singular, adding regularization.")
                print('voxel =', voxel)
                print_matrix(Omega, 'Omega')
                print_matrix(h, 'h')
                try:
                    Omega += min_omega_diag * np.eye(k)
                    inv_term = inv(Omega) + (1/Sigma2i) * (Xi.T @ Xi)
                except np.linalg.LinAlgError:
                    print("Matrix is still singular, skipping this iteration.")
                    continue
            
            H[i] = inv(inv_term)
            h[i] = H[i] @ (inv(Omega) @ Mu[i] + (1/Sigma2i) * Xi.T @ yi)
        
        # M-шаг: обновление B, Omega, sigma_sq
        B_new = inv(X_star.T @ X_star) @ X_star.T @ h
        
        Mu_new = np.zeros((M, k))
        for i in range(M):
            Mu_new[i] = B.T @ X_star[i]
        Mu_sum_new = np.sum(Mu, axis=0)

        Omega_new = np.zeros((k, k))
        for i in range(M):
            diff = h[i] - B_new.T @ X_star[i]
            #Omega_new += np.outer(diff, diff) + H[i]
            Omega_new += H[i] + h[i] @ h[i].T
        Omega_new -= (Mu_sum @ Mu_sum.T) / M
        #Omega_new /= M
        Omega_new /= (M - k_star)
        Omega_new += np.eye(k) * min_omega_diag  # Регуляризация для устойчивости

        Sigma2_new = np.zeros(M)
        for i in range(M):
            #Sigma2_new[i] = np.mean((Y_list[i] - X_list[i] @ h[i])**2) + np.trace(X_list[i] @ H[i] @ X_list[i].T)
            #Sigma2_new[i] = np.linalg.norm(Y_list[i] - X_list[i] @ h[i])**2 / (n - k) + np.trace(X_list[i] @ H[i] @ X_list[i].T)
            Sigma2_new[i] = np.linalg.norm(Y_list[i] - X_list[i] @ h[i])**2 / (n - k)

        if np.max(np.abs(B_new - B)) < tol and \
           np.max(np.abs(Omega_new - Omega)) < tol and \
           np.max(np.abs(Sigma2_new - Sigma2)) < tol:
            break
        B, Omega, Sigma2, Mu, Mu_sum = B_new, Omega_new, Sigma2_new, Mu_new, Mu_sum_new
        h = h.copy()
        H = H.copy()
        Sigma2 = np.maximum(Sigma2_new, min_sigma2)  # Защита от нулевой дисперсии
        del Mu_new, Mu_sum_new, B_new, Omega_new, Sigma2_new
        # Проверка на NaN
        if np.isnan(h).any() or np.isnan(H).any() or np.isnan(Omega).any():
            print("NaN values detected, restarting EM algorithm.")
            h = np.zeros((M, k))
            H = np.zeros((M, k, k))
            Omega = np.eye(k) * 0.1
            Sigma2 = np.ones(M) * min_sigma2
            Mu = np.zeros((M, k))
            continue
        '''
        # Проверка на вырожденность
        if np.linalg.cond(Omega) > 1 / min_sigma2:
            print("Matrix Omega is ill-conditioned, restarting EM algorithm.")
            h = np.zeros((M, k))
            H = np.zeros((M, k, k))
            Omega = np.eye(k) * 0.1
            Sigma2 = np.ones(M) * min_sigma2
            Mu = np.zeros((M, k))
            continue
        '''
    Mu_sum = np.sum(Mu, axis=0)
    return h, H, Mu_sum, Omega

def run_EM(X_list, Y_list, X_star, nX=None, nY=None, nZ=None, nR=None, nT=None, nS=None, \
           max_iter=100, tol=1e-4, min_sigma2=1e-10):
    t1 = time.time()
    #print('Матрицы для EM-алгоритма:')
    #print('Матрицы X', X_matrixes.shape)
    #print(X_matrixes)
    #print('Матрицы Y', Y_matrixes.shape)
    #print(Y_matrixes)
    #Изначально Гиперпараметры Omega, Sigma2, Mu, X* - просто какие-то
    h = np.zeros((nR, nS)) #прогон стимул
    H = np.zeros((nR, nS, nS)) #прогон стимул^2
    #Z = np.zeros(())
    for r in range(nR):
        for s in range(nS):
            H[r][s, s] = 1
    
    k_star = X_star.shape[1]
    M = nR
    k = nS
    n = nT # пока так, не очень понятно, что такое n
    #Берём простой вид Omega, Sigma2 и Mu
    Omega = np.eye(nS) # ковариационная матрица параметров регрессии
    Sigma2 = np.ones(nR)
    Mu = np.ones((nR, nS)) #для каждого прогона это вектор средних параметров регрессии
    curr_vars = {'Omega':Omega, 'Sigma2':Sigma2, 'Mu':Mu, 'h':h, 'H':H}
    prev_vars = {'Omega':Omega, 'Sigma2':Sigma2, 'Mu':Mu, 'h':h, 'H':H}
    #X_i - матрица регрессии, задана заранее
    #X2 задана заранее = X*
    #k2 задано заранее = k*
    n_iter = 0
    while ((max2(curr_vars['h'] - prev_vars['h']) > tol) \
    and (max3(curr_vars['H'] - prev_vars['H']) > tol) and \
         any(curr_vars['Sigma2'] > tol) and n_iter < max_iter) or (n_iter < 2):
        # запоминаем 'старые' оценки значений
        prev_vars['Omega'] = copy.deepcopy(curr_vars['Omega'])
        prev_vars['Sigma2'] = copy.deepcopy(curr_vars['Sigma2'])
        prev_vars['Mu'] = copy.deepcopy(curr_vars['Mu'])
        prev_vars['h'] = copy.deepcopy(curr_vars['h'])
        prev_vars['H'] = copy.deepcopy(curr_vars['H'])
        for i in range(nR):#по всем прогонам
            Xi = X_list[i]
            yi = Y_list[i]
            Mui = prev_vars['Mu'][i]
            Sigma2i = prev_vars['Sigma2'][i]
            #print(prev_vars['Omega'])
            invOmega = np.linalg.inv(prev_vars['Omega'])
            if check_nan2(invOmega):
                print('NaN in invOmega')
                invOmega = np.linalg.pinv(prev_vars['Omega'])
            
            #Оценка среднего
            #print(invOmega.shape, Sigma2i.shape, Xi.shape, Mui.shape, yi.shape)
            #print(Sigma2i)
            #print(invOmega)
            #print(Xi.T @ Xi)
            #print(invOmega + (1/Sigma2i) * (Xi.T @ Xi))
            H[i] = np.linalg.inv(invOmega + 1 / Sigma2i * (Xi.T @ Xi))
            if check_nan2(H[i]):
                print('NaN in H')
                #H[i] = prev_vars['H'][i]
                H[i] = np.linalg.pinv(invOmega + 1 / Sigma2i * (Xi.T @ Xi))
            h[i] = H[i] @ (invOmega @ Mui + (1/Sigma2i) * Xi.T @ yi)
            if check_nan1(h[i]):
                print('NaN in h')
                #h[i] = prev_vars['h'][i]
                h[i] = H[i] @ (invOmega @ Mui + (1/Sigma2i) * Xi.T @ yi)

            
        
        curr_vars['h'] = copy.deepcopy(h)
        curr_vars['H'] = copy.deepcopy(H)

        #M-шаг
        B = np.linalg.solve(X_star.T @ X_star, X_star.T @ prev_vars['h']) #B - это k* на k
        #B = np.linalg.inv(X_star.T @ X_star) @ X_star.T @ prev_vars['h'] #здесь не до конца уверен, что берем именно предыдущую оценку для h
        if check_nan2(B):
            print('NaN in B')
            B = np.linalg.pinv(X_star.T @ X_star) @ X_star.T @ prev_vars['h']
        for i in range(nR):
            Mu[i] = B.T @ X_star[i, :]
        curr_vars['Mu'] = copy.deepcopy(Mu)
        Mu_sum = sum(prev_vars['Mu']) #тут тоже не 100% уверенность, что используем предудущие
        
        Omega = (1/(M-k_star) * (sum([prev_vars['H'][i] + prev_vars['h'][i] @ prev_vars['h'][i].T \
                                  for i in range(nR)]) - (1/M) * Mu_sum @ Mu_sum.T)  )
        curr_vars['Omega'] = copy.deepcopy(Omega)
        for i in range(nR):
            yi = Y_list[i]
            Xi = X_list[i]
            hi = prev_vars['h'][i]
            Sigma2[i] = max((1/(n-k)) * MSE(yi, Xi @ hi), min_sigma2)
             #тут нет уверенности, что такое n
        curr_vars['Sigma2'] = copy.deepcopy(Sigma2)

        n_iter += 1
    t2 = time.time()
    print(f'Остановились на {n_iter} итерациях')
    print(f'За время {t2-t1} секунд')
    print(f'avg_diff2 = {avg_diff2(curr_vars['h'], prev_vars['h'])}')
    print(f'max h diff = {max2(curr_vars['h'] - prev_vars['h'])}')
    print(f'max H diff = {max3(curr_vars['H'] - prev_vars['H'])}')
    print(f'avg_diff3 = {avg_diff3(curr_vars['H'], prev_vars['H'])}')
    print(f'max_sigma2_diff = {max(curr_vars['Sigma2'] - prev_vars['Sigma2'])}')
    #print(curr_vars['h'])
    #print(curr_vars['H'])
    #print(curr_vars['Sigma2'])
    return curr_vars['h'], curr_vars['H']

def max2(a):
    n1, n2 = a.shape
    genmax = -np.inf
    for i in range(n1):
        for j in range(n2):
            if abs(a[i, j]) > genmax:
                genmax = abs(a[i, j])
    return genmax

def max3(a):
    n1, n2, n3 = a.shape
    genmax = -np.inf
    for i in range(n1):
        for j in range(n2):
            for k in range(n3):
                if abs(a[i, j, k]) > genmax:
                    genmax = abs(a[i, j, k])
    return genmax

def check_nan1(a):
    n = a.shape[0]
    for i in range(n):
        if np.isnan(a[i]):
            return True
    return False

def check_nan2(a):
    n1, n2 = a.shape
    for i in range(n1):
        for j in range(n2):
            if np.isnan(a[i, j]):
                return True
    return False

def check_nan3(a):
    n1, n2, n3 = a.shape
    for i in range(n1):
        for j in range(n2):
            for k in range(n3):
                if np.isnan(a[i, j, k]):
                    return True
    return False

def norm2(a):
    n1, n2 = a.shape
    sum = 0
    for i in range(n1):
        for j in range(n2):
            sum += a[i, j]**2
    return sum


def run_regression(patient_data_path=None, patient_logs_path=None, voxel=None, X2=None, voxel_data=None, nX=None, nY=None, nZ=None, nR=None, nT=None, nS=None):
    if voxel_data == None:
        t1 = time.time()
        X_list = get_X(patient_logs_path, voxel, nX, nY, nZ, nR, nT, nS) #матрицы дизайна для прогонов
        print('Матрица дизайна:\n', X_list[0])
        t2 = time.time()
        Y_list = get_Y(patient_data_path, voxel) #столбец состояния вокселя на сканах для прогонов
        print('Столбец интенсивности вокселя:\n', Y_list[0])
        t3 = time.time()
        mean_beta, disp_beta = em_algorithm_glm(X_list, Y_list, X2, nX, nY, nZ, nR, nT, nS, sparse=True)
        t4 = time.time()
        #print('Матрица X:', t2-t1)
        #print('Матрица Y:', t3-t2)
        #print('EM-алгоритм:', t4-t3)
        return mean_beta, disp_beta


def gauss_smooth(data):
    new_data = scipy.ndimage.gaussian_filter(input=data, radius=0.8)
    return new_data


def get_t_value(mean_beta, disp_beta, contrast, nX=None, nY=None, nZ=None, nR=None, nT=None, nS=None):
    #disp_beta = gauss_smooth(disp_beta)
    t_value_list = np.array([np.sum(np.array([mean_beta[i][j] * contrast[j] for j in range(len(contrast))])) \
                             / np.sqrt(np.sum(np.array([[contrast[j]*contrast[k]*disp_beta[i][j][k] for j in range(len(contrast))] \
                                           for k in range(len(contrast))]))) for i in range(len(mean_beta))])
    return t_value_list


def get_T_value(mu, Omega, contrast,  nX=None, nY=None, nZ=None, nR=None, nT=None, nS=None):
    #disp_beta = gauss_smooth(disp_beta)
    T_value = np.sum([mu[i] * contrast[i] for i in range(len(contrast))]) / \
        np.sqrt(np.sum([[contrast[i] * contrast[j] * Omega[i, j] for j in range(len(contrast))] for i in range(len(contrast))]))
    return T_value



def voxel_coord(v, nX, nY, nZ):
    return (v % (nX * nY)) % nY, (v % (nX * nY)) // nY, v // (nX * nY)


def coord_voxel(x, y, z, nX, nY, nZ):
    return nX * nY * z + nX * y + x



def get_thresholds_dict(K_list, t_values, nX=None, nY=None, nZ=None, nR=None, nT=None, nS=None, mode='value'):
    t_values_list = []
    for v in range(t_values.shape[0]):
        for r in range(t_values.shape[1]):
            t_values_list.append(t_values[v, r])
    t_values_list_sorted = sorted(t_values_list)

    min_value = min(t_values_list)
    max_value = max(t_values_list)
    dict_thresholds = {}
    #dict_thresholds['numb'] = {}
    #dict_thresholds['value'] = {}

    for K in K_list:
        n_elem = t_values.shape[0] * t_values.shape[1] // (K+1)
        dict_thresholds[K] = []
        #dict_thresholds['numb'][K] = []
        #dict_thresholds['value'][K] = []
        #dict_random_thresholds[K] = []
        for i in range(1, K+1):
            if mode == 'numb':
                dict_thresholds[K].append(t_values_list_sorted[i * n_elem])
            if mode == 'value':
                dict_thresholds[K].append(min_value + (max_value - min_value) * i / K)
        #plt.hist(t_values_list_sorted, bins=100)
        #for i in range(0, K):
        #    plt.axvline(x=dict_thresholds[K][i], color='r', linestyle='--')
    return dict_thresholds


def get_t_groups(K_list, dict_thresholds, t_value, nX=None, nY=None, nZ=None, nR=None, nT=None, nS=None, mode='value'):
    nV = nX * nY * nZ
    t_value_voxel_groups = {}
    #t_value_voxel_groups['numb'] = {}
    #t_value_voxel_groups['value'] = {}
    for K in K_list:
        t_value_voxel_groups[K] = np.zeros((nV, K+1))
        #t_value_voxel_groups['numb'][K] = np.zeros((nV, K+1))
        #t_value_voxel_groups['value'][K] = np.zeros((nV, K+1))
        for v in range(t_value.shape[0]):
            for n_group in range(0, K+1):
                if n_group == 0:
                    
                    #Нулевая группа - все, что меньше нулевого порога
                    t_value_voxel_groups[K][v][n_group] = len([t for t in t_value[v] \
                                                                    if (t < dict_thresholds[K][n_group])])
                    #t_value_voxel_groups['numb'][K][v][n_group] = len([t for t in t_value[v] \
                    #                                                 if (t < dict_thresholds['numb'][K][n_group])])
                    #t_value_voxel_groups['value'][K][v][n_group] = len([t for t in t_value[v] \
                    #                                                  if (t < dict_thresholds['value'][K][n_group])])
                elif n_group == K:
                    
                    #K-я группа - все, что больше (K-1)-го порога
                    t_value_voxel_groups[K][v][n_group] = len([t for t in t_value[v] \
                                                                     if (t >= dict_thresholds[K][n_group - 1])])
                    #t_value_voxel_groups['numb'][K][v][n_group] = len([t for t in t_value[v] \
                    #                                                 if (t > dict_thresholds['numb'][K][n_group - 1])])
                    #t_value_voxel_groups['value'][K][v][n_group] = len([t for t in t_value[v] \
                    #                                                  if (t > dict_thresholds['value'][K][n_group - 1])])
                else:
                    
                    #i-я группа - от (i-1)-го до i-го порога
                    t_value_voxel_groups[K][v][n_group] = len([t for t in t_value[v] \
                                                                     if (t >= dict_thresholds[K][n_group - 1] \
                                                                         and t < dict_thresholds[K][n_group])])
                    #t_value_voxel_groups['numb'][K][v][n_group] = len([t for t in t_value[v] \
                    #                                                 if (t > dict_thresholds['numb'][K][n_group - 1] \
                    #                                                     and t < dict_thresholds['numb'][K][n_group])])
                    #t_value_voxel_groups['value'][K][v][n_group] = len([t for t in t_value[v] \
                    #                                                  if (t > dict_thresholds['value'][K][n_group - 1] \
                    #                                                      and t < dict_thresholds['value'][K][n_group])])
    return t_value_voxel_groups #Это значения r, которые нужны для функции правдоподобия




def f_obj(lamda, P_A, P_I, t_values):
    r = t_values
    K = P_A.shape[0]
    V = t_values.shape[0]
    f = np.sum([np.log(lamda * \
                   np.prod([P_A[k]**r[v,k] for k in range(K)]) + \
                   (1 - lamda) * np.prod([P_I[k]**r[v,k] for k in range(K)])) \
                   for v in range(V)])
    return f 

def f_obj3(lamda, P_A, P_I, t_values):
    r = t_values
    K = P_A.shape[0]
    V = t_values.shape[0]
    f = np.prod([(lamda * \
                   np.prod([P_A[k]**r[v,k] for k in range(K)]) + \
                   (1 - lamda) * np.prod([P_I[k]**r[v,k] for k in range(K)])) \
                   for v in range(V)])
    return f 







def J_obj(lamda, P_A, P_I, t_values):
    r = t_values
    K = P_A.shape[0] - 1
    V = t_values.shape[0]
    J = np.zeros(K * 2 + 3)
    #print(r.shape, K, V)
    J[0] = np.prod([np.prod([P_A[k]**r[v,k] for k in range(K)]) - \
                    np.prod([P_I[k]**r[v,k] for k in range(K)])\
                    for v in range(V)])
    for i in range(1, K+2):
        ind = (i-1)%(K+1)
        J[i] = np.prod([
                        lamda*\
                        np.prod([P_A[k]**r[v,k] for k in [j for j in range(K) if j!=ind]])*\
                       r[v,ind]*P_A[ind]**(r[v,ind]-1)\
                        for v in [j for j in range(V) if r[j,ind]>=1]
                        ])
    for i in range(K+2, 2*K+3):
        ind = (i-1)%(K+1)
        J[i] = np.prod([(1-lamda)*\
                        np.prod([P_I[k]**r[v,k] for k in [j for j in range(K) if j!=ind]])*\
                       r[v,ind]*P_I[ind]**(r[v,ind]-1)\
                       for v in [j for j in range(V) if r[j,ind]>=1]])
    return J





def H_obj(lamda, P_A, P_I, t_values):
    r = t_values
    K = P_A.shape[0]-1
    V = t_values.shape[0]
    H = np.zeros((K * 2 + 3, 2 * K + 3))
    #Элемент (0,0)
    H[0,0] = 0
    #0-й столбец P_A_K
    for i in range(1, K+2):
        ind = (i-1)%(K+1)
        H[0,i] = np.prod([\
                        np.prod([P_A[k]**r[v,k] for k in [l for l in range(K) if l!=ind]])*\
                       r[v,ind]*P_A[ind]**(r[v,ind]-1)\
                       for v in [l for l in range(V) if r[l,ind]>=1]])
    #0-й столбцеw P_I_K
    for i in range(K+2, 2*K+3):
        ind = (i-1)%(K+1)
        H[0,i] = np.prod([-1*\
                        np.prod([P_I[k]**r[v,k] for k in [l for l in range(K) if l!=ind]])*\
                       r[v,ind]*P_I[ind]**(r[v,ind]-1)\
                       for v in [l for l in range(V) if r[l,ind]>=1]])
    #Главная диагональ P_A_k
    for i in range(1, K+2):
        ind = (i-1)%(K+1)
        H[i,i] = np.prod([lamda*\
                        np.prod([P_A[k]**r[v,k] for k in [l for l in range(K) if l!=ind]])*\
                       r[v,ind]*(r[v,ind]-1)*P_A[ind]**(r[v,ind]-2)\
                       for v in [l for l in range(V) if r[l,ind]>=2]])
    #Главная диагональ P_I_k
    for i in range(K+2, 2*K+3):
        ind = (i-1)%(K+1)
        H[i,i] = np.prod([(1-lamda)*\
                        np.prod([P_I[k]**r[v,k] for k in [l for l in range(K) if l!=ind]])*\
                       r[v,ind]*(r[v,ind]-1)*P_I[ind]**(r[v,ind]-2)\
                       for v in [l for l in range(V) if r[l,ind]>=2]])
    #P_A_i P_A_j ниже главной диагонали
    for i in range(1, K+2):
        for j in range(1, i):
            ind1 = (i-1)%(K+1)
            ind2 = (j-1)%(K+1)
            H[i,j] = np.prod([lamda*\
                        np.prod([P_A[k]**r[v,k] for k in [l for l in range(K) if (l!=ind1) and (l!=ind2)]])*\
                       r[v,ind1]*P_A[ind1]**(r[v,ind1]-1)*r[v,ind2]*P_A[ind2]**(r[v,ind2]-1)\
                       for v in [l for l in range(V) if (r[l,ind1]>=1) and (r[l,ind2]>=1)]])
    #P_A_i P_A_j ниже главной диагонали
    for i in range(K+2, 2*K+3):
        for j in range(K+2, i):
            ind1 = (i-1)%(K+1)
            ind2 = (j-1)%(K+1)
            H[i,j] = np.prod([(1-lamda)*\
                        np.prod([P_I[k]**r[v,k] for k in [l for l in range(K) if (l!=ind1) and (l!=ind2)]])*\
                       r[v,ind1]*P_I[ind1]**(r[v,ind1]-1)*r[v,ind2]*P_I[ind2]**(r[v,ind2]-1)\
                       for v in [l for l in range(V) if (r[l,ind1]>=1) and (r[l,ind2]>=1)]])
    #P_I_i P_A_j
    for i in range(K+2, 2*K+3):
        for j in range(1, K+2):
            H[i,j] = 0
    #Симметрия
    for i in range(2*K+3):
        for j in range(i+1, 2*K+3):
            H[i,j] = H[j,i]
    return H


def f_obj_2(lamda, P_A, P_I, t_values):
    r = t_values
    n_t_dict = {}
    K = P_A.shape[0]-1
    V = t_values.shape[0]

    for v in range(V):
        t_numbers_list = tuple(r[v])
        if t_numbers_list not in n_t_dict.keys():
            n_t_dict[t_numbers_list] = 0
        n_t_dict[t_numbers_list] += 1

    f = np.sum([n_t_dict[t_numbers_list] * np.log(lamda * \
                    np.prod([P_A[k]**t[k] for k in range(len(P_A))]) + (1-lamda) * \
                   np.prod([P_I[k]**t[k] for k in range(len(P_A))])) \
                             for t in n_t_dict.keys()])
    
    #f = - np.prod([lamda * \
    #               np.prod([P_A[k]**r[v,k] for k in range(K)]) + \
    #               (1 - lamda) * np.prod([P_I[k]**r[v,k] for k in range(K)]) \
    #               for v in range(V)])
    return f 



def denominator(lamda, P_A, P_I, t):
    K = P_A.shape[0] - 1
    
    return lamda * np.prod([P_A[k]**t[k] for k in range(K)]) + \
    (1-lamda) * np.prod([P_I[k]**t[k] for k in range(K)])





def J_obj_2(lamda, P_A, P_I, t_values):
    r = t_values
    n_t_dict = {}
    K = P_A.shape[0]-1
    V = t_values.shape[0]

    for v in range(V):
        t_numbers_list = tuple(r[v])
        if t_numbers_list not in n_t_dict.keys():
            n_t_dict[t_numbers_list] = 0
        n_t_dict[t_numbers_list] += 1

    J = np.zeros(K * 2 + 3)
    #print(r.shape, K, V)
    J[0] = np.sum([n_t_dict[t] * (np.prod([P_A[k]**t[k] for k in range(K)]) - \
            np.prod([P_I[k]**t[k] for k in range(K)])) / denominator(lamda, P_A, P_I, t)\
                    for t in n_t_dict.keys()])
    for i in range(1, K+2):
        ind = (i-1)%(K+1)
        J[i] = np.sum([n_t_dict[t] * (lamda *\
                        np.prod([P_A[k]**t[k] for k in range(K) if k!=ind]) *\
                       t[ind] * P_A[ind]**(t[ind]-1) )/ denominator(lamda, P_A, P_I, t)\
                        for t in n_t_dict.keys() if t[ind]>=1])
    for i in range(K+2, 2*K+3):
        ind = (i-1)%(K+1)
        J[i] = np.sum([n_t_dict[t] * ((1-lamda) *
                        np.prod([P_I[k]**t[k] for k in range(K) if k!=ind])*\
                       t[ind] * P_I[ind]**(t[ind]-1)) / denominator(lamda, P_A, P_I, t)\
                        for t in n_t_dict.keys() if t[ind]>=1])
    return J


def H_obj_2(lamda, P_A, P_I, t_values):
    r = t_values
    n_t_dict = {}
    K = P_A.shape[0]-1
    V = t_values.shape[0]

    for v in range(V):
        t_numbers_list = tuple(r[v])
        if t_numbers_list not in n_t_dict.keys():
            n_t_dict[t_numbers_list] = 0
        n_t_dict[t_numbers_list] += 1

    H = np.zeros((2 * K + 3, 2 * K + 3))
    #Элемент (0,0)
    H[0,0] = -np.sum([n_t_dict[t] * (np.prod([P_A[k]**t[k] for k in range(K)]) -\
                          np.prod([P_I[k]**t[k] for k in range(K)]))**2 / denominator(lamda, P_A, P_I, t)**2\
                       for t in n_t_dict.keys()])
    
    #0-й столбец P_A_K
    for i in range(1, K+2):
        ind = (i-1)%(K+1)
        H[0,i] = np.sum([n_t_dict[t] * (np.prod([P_A[k]**t[k] for k in range(K) if k!=ind]) *\
                       t[ind] * P_A[ind]**(t[ind]-1) * denominator(lamda, P_A, P_I, t) - \
                       lamda * np.prod([P_A[k]**t[k] for k in range(K) if k!=ind]) * t[ind] * P_A[ind]**(t[ind]-1) * \
                       (np.prod([P_A[k]**t[k] for k in range(K)]) - np.prod([P_I[k]**t[k] for k in range(K)]))) / denominator(lamda, P_A, P_I, t)**2\
                       for t in n_t_dict.keys() if t[ind]>=1])
        
    #0-й столбцеw P_I_K
    for i in range(K+2, 2*K+3):
        ind = (i-1)%(K+1)
        H[0,i] = np.sum([n_t_dict[t] * (-np.prod([P_I[k]**t[k] for k in range(K) if k!=ind]) *\
                       t[ind] * P_I[ind]**(t[ind]-1) * denominator(lamda, P_A, P_I, t) - \
                       (1-lamda) * np.prod([P_I[k]**t[k] for k in range(K) if k!=ind]) * t[ind] * P_I[ind]**(t[ind]-1) * \
                       (np.prod([P_A[k]**t[k] for k in range(K)]) - np.prod([P_I[k]**t[k] for k in range(K)]))) / denominator(lamda, P_A, P_I, t)**2\
                       for t in n_t_dict.keys() if t[ind]>=1])
        
    #Главная диагональ P_A_k
    for i in range(1, K+2):
        ind = (i-1)%(K+1)
        H[i,i] = np.sum([n_t_dict[t] * (lamda *\
                        np.prod([P_A[k]**t[k] for k in range(K) if k!=ind]) *\
                       t[ind]*(t[ind]-1)*P_A[ind]**(t[ind]-2) * denominator(lamda, P_A, P_I, t) - \
           (lamda * np.prod([P_A[k]**t[k] for k in range(K) if k!=ind]) * t[ind] * P_A[ind]**(t[ind]-1))**2) / denominator(lamda, P_A, P_I, t)**2\
                       for t in n_t_dict.keys() if t[ind]>=1])
        
    #Главная диагональ P_I_k
    for i in range(K+2, 2*K+3):
        ind = (i-1)%(K+1)
        H[i,i] = np.sum([n_t_dict[t] * ((1-lamda) *\
                        np.prod([P_I[k]**t[k] for k in range(K) if k!=ind]) *\
                       t[ind]*(t[ind]-1)*P_I[ind]**(t[ind]-2) * denominator(lamda, P_A, P_I, t) - \
           ((1-lamda) * np.prod([P_I[k]**t[k] for k in range(K) if k!=ind]) * t[ind] * P_I[ind]**(t[ind]-1))**2) / denominator(lamda, P_A, P_I, t)**2\
                       for t in n_t_dict.keys() if t[ind]>=1])
        
    #P_A_i P_A_j ниже главной диагонали
    for i in range(1, K+2):
        for j in range(1, i):
            ind1 = (i-1)%(K+1)
            ind2 = (j-1)%(K+1)
            H[i,j] = np.sum([n_t_dict[t] * (lamda *\
                        np.prod([P_A[k]**t[k] for k in range(K) if (k!=ind1) and (k!=ind2)])*\
                       t[ind1]*P_A[ind1]**(t[ind1]-1)*t[ind2]*P_A[ind2]**(t[ind2]-1) * denominator(lamda, P_A, P_I, t) - \
                       (lamda * np.prod([P_A[k]**t[k] for k in range(K) if k!=ind1]) * t[ind1] * P_A[ind1]**(t[ind1]-1)) * \
                       (lamda * np.prod([P_A[k]**t[k] for k in range(K) if k!=ind2]) * t[ind2] * P_A[ind2]**(t[ind2]-1))) / \
                              denominator(lamda, P_A, P_I, t)**2\
                       for t in n_t_dict.keys() if (t[ind1]>=1) and (t[ind2]>=1)])
            
    #P_I_i P_I_j ниже главной диагонали
    for i in range(K+2, 2*K+3):
        for j in range(K+2, i):
            ind1 = (i-1)%(K+1)
            ind2 = (j-1)%(K+1)
            H[i,j] = np.sum([n_t_dict[t] * ((1-lamda) *\
                        np.prod([P_I[k]**t[k] for k in range(K) if (k!=ind1) and (k!=ind2)]) *\
                       t[ind1]*P_I[ind1]**(t[ind1]-1)*t[ind2]*P_I[ind2]**(t[ind2]-1) * denominator(lamda, P_A, P_I, t) - \
                       ((1-lamda) * np.prod([P_I[k]**t[k] for k in range(K) if k!=ind1]) * t[ind1] * P_I[ind1]**(t[ind1]-1)) * \
                       ((1-lamda) * np.prod([P_I[k]**t[k] for k in range(K) if k!=ind2]) * t[ind2] * P_I[ind2]**(t[ind2]-1))) / \
                              denominator(lamda, P_A, P_I, t)**2\
                       for t in n_t_dict.keys() if (t[ind1]>=1) and (t[ind2]>=1)])
            
    #P_I_j P_A_i
    for j in range(K+2, 2*K+3):
        for i in range(1, K+2):
            ind1 = (i-1)%(K+1)
            ind2 = (j-1)%(K+1)
            H[i,j] = np.sum([n_t_dict[t] * (lamda * np.prod([P_A[k]**t[k] for k in range(K) if k!=ind1]) * t[ind1] * P_A[ind1]**(t[ind1]-1) * \
        lamda * np.prod([P_I[k]**t[k] for k in range(K) if k!=ind2]) * t[ind2] * P_I[ind2]**(t[ind2]-1)) / denominator(lamda, P_A, P_I, t)**2 \
                            for t in n_t_dict.keys() if t[ind1]>=1 and t[ind2]>=1])

    #Симметрия
    for i in range(2*K+3):
        for j in range(i+1, 2*K+3):
            H[i,j] = H[j,i]
    return H


def constraint1(K):
    Matr1 = [[1 if j == (i + 1) else 0 for j in range(2*K+3)] for i in range(K+1)]
    LB1 = [0 for i in range(K+1)]
    UB1 = [1 for i in range(K+1)]
    return Matr1, LB1, UB1







def constraint2(K):
    Matr2 = [[1 if j == (i + K + 2) else 0 for j in range(2*K+3)] for i in range(K+1)]
    LB2 = [0 for i in range(K+1)]
    UB2 = [1 for i in range(K+1)]
    return Matr2, LB2, UB2




def constraint3(K):
    Matr3 = [[1 if i >= 1 and i <= (K+1) else 0 for i in range(2*K+3)]]
    LB3 = [-np.inf]
    UB3 = [1]
    return Matr3, LB3, UB3




def constraint32(K):
    Matr3 = [[1 if i >= 1 and i <= (K+1) else 0 for i in range(2*K+3)]]
    LB3 = [1]
    UB3 = [np.inf]
    return Matr3, LB3, UB3





def constraint4(K):
    Matr4 = [[1 if i >= (K+2) and i <= (2*K+2) else 0 for i in range(2*K+3)]]
    LB4 = [1]
    UB4 = [np.inf]
    return Matr4, LB4, UB4 


def constraint42(K):
    Matr4 = [[1 if i >= (K+2) and i <= (2*K+2) else 0 for i in range(2*K+3)]]
    LB4 = [-np.inf]
    UB4 = [1]
    return Matr4, LB4, UB4 





def constraint5(K):
    Matr5 = [1 if i == 0 else 0 for i in range(2*K+3)]
    LB5 = [0]
    UB5 = [1]
    return Matr5, LB5, UB5




def get_constraints(K):
    Matr1, LB1, UB1 = constraint1(K)
    const1 = LinearConstraint(Matr1, LB1, UB1)
    Matr2, LB2, UB2 = constraint2(K)
    const2 = LinearConstraint(Matr2, LB2, UB2)
    Matr3, LB3, UB3 = constraint3(K)
    const3 = LinearConstraint(Matr3, LB3, UB3)
    Matr4, LB4, UB4 = constraint4(K)
    const4 = LinearConstraint(Matr4, LB4, UB4)
    Matr5, LB5, UB5 = constraint5(K)
    const5 = LinearConstraint(Matr5, LB5, UB5)

    Matr32, LB32, UB32 = constraint32(K)
    const32 = LinearConstraint(Matr32, LB32, UB32)
    Matr42, LB42, UB42 = constraint42(K)
    const42 = LinearConstraint(Matr42, LB42, UB42)
    return const1, const2, const3, const4, const5, const32, const42






def solve_prob(K, t_value_voxel_groups, nX=None, nY=None, nZ=None, \
        nR=None, nT=None, nS=None, f_func=f_obj_2, J_func=J_obj_2, H_func=H_obj_2, lamda=None, way='lamda', mode='value'):
    print(way)
    t_values = t_value_voxel_groups[K]
    #t_values_numb = t_value_voxel_groups[K]
    #t_values_value = t_value_voxel_groups[K]
    const1, const2, const3, const4, const5, const32, const42 = get_constraints(K)
    if lamda is None:
        lamda = 0.02
    variables_list = []
    variables_list.append(0.5)
    variables_list.extend([1/K for _ in range(2*K + 2)])
    const_list = [const1, const2, const3, const4, const5, const32, const42]
    if way != 'lamda':
        print('lambda is variable')
        res = scipy.optimize.minimize(lambda vector: \
                                           -f_func(lamda=vector[0], P_A=vector[1:(K+2)], P_I=vector[(K+2):(2*K+3)], t_values=t_values), variables_list, \
                                           method='trust-constr', constraints=const_list, jac=lambda vector: \
                                           -J_func(lamda=vector[0], P_A=vector[1:(K+2)], P_I=vector[(K+2):(2*K+3)], t_values=t_values)\
                                           , hess=lambda vector: \
                                           -H_func(lamda=vector[0], P_A=vector[1:(K+2)], P_I=vector[(K+2):(2*K+3)], t_values=t_values), options={'verbose': 1})
        #res_numb = scipy.optimize.minimize(lambda x: \
        #                                   -f(lamda=x[0], P_I=x[1:(K+2)], P_A=x[(K+2):(2*K+3)], t_values=t_values_numb), variables_list, \
        #                                   method='trust-constr', constraints=const_list, jac=lambda x: \
        #                                   -J(lamda=x[0], P_I=x[1:(K+2)], P_A=x[(K+2):(2*K+3)], t_values=t_values_numb)\
        #                                   , hess=lambda x: \
        #                                   -H(lamda=x[0], P_I=x[1:(K+2)], P_A=x[(K+2):(2*K+3)], t_values=t_values_numb), options={'verbose': 1})
        #res_value = scipy.optimize.minimize(lambda x: \
        #                                   -f(lamda=x[0], P_I=x[1:(K+2)], P_A=x[(K+2):(2*K+3)], t_values=t_values_value), variables_list, \
        #                                    method='trust-constr', constraints=const_list, jac=lambda x: \
        #                                   -J(lamda=x[0], P_I=x[1:(K+2)], P_A=x[(K+2):(2*K+3)], t_values=t_values_value)\
        #                                    , hess=lambda x: \
        #                                   -H(lamda=x[0], P_I=x[1:(K+2)], P_A=x[(K+2):(2*K+3)], t_values=t_values_value), options={'verbose': 1})
    else:
        print('fixed lambda =', lamda)
        res = scipy.optimize.minimize(lambda vector: \
                                           -f_func(lamda=lamda, P_A=vector[1:(K+2)], P_I=vector[(K+2):(2*K+3)], t_values=t_values), variables_list, \
                                           method='trust-constr', constraints=const_list, jac=lambda vector: \
                                           -J_func(lamda=lamda, P_A=vector[1:(K+2)], P_I=vector[(K+2):(2*K+3)], t_values=t_values)\
                                           , hess=lambda vector: \
                                           -H_func(lamda=lamda, P_A=vector[1:(K+2)], P_I=vector[(K+2):(2*K+3)], t_values=t_values), options={'verbose': 1})
        #res_numb = scipy.optimize.minimize(lambda x: \
        #                                   -f(lamda=lamda, P_I=x[1:(K+2)], P_A=x[(K+2):(2*K+3)], t_values=t_values_numb), variables_list, \
        #                                   method='trust-constr', constraints=const_list, jac=lambda x: \
        #                                   -J(lamda=lamda, P_I=x[1:(K+2)], P_A=x[(K+2):(2*K+3)], t_values=t_values_numb)\
        #                                   , hess=lambda x: \
        #                                  -H(lamda=lamda, P_I=x[1:(K+2)], P_A=x[(K+2):(2*K+3)], t_values=t_values_numb), options={'verbose': 1})
        #res_value = scipy.optimize.minimize(lambda x: \
        #                                   -f(lamda=lamda, P_I=x[1:(K+2)], P_A=x[(K+2):(2*K+3)], t_values=t_values_value), variables_list, \
        #                                    method='trust-constr', constraints=const_list, jac=lambda x: \
        #                                   -J(lamda=lamda, P_I=x[1:(K+2)], P_A=x[(K+2):(2*K+3)], t_values=t_values_value)\
        #                                    , hess=lambda x: \
        #                                   -H(lamda=lamda, P_I=x[1:(K+2)], P_A=x[(K+2):(2*K+3)], t_values=t_values_value), options={'verbose': 1})
    return res



def print_get_solution(K, t_value_voxel_groups, need_print=True, nX=None, nY=None, nZ=None,\
                       nR=None, nT=None, nS=None, mode='value', way='lamda', lamda=None):
    lambda_, P_A_list, P_I_list = EM_optimization(K=K, r=t_value_voxel_groups, nX=nX, nY=nY, nZ=nZ, nR=nR, nT=nT, nS=nS, \
                     mode=way)
    
    #variables = res.x
    #lamda = variables[0]
    #P_A_list = variables[1:(K+2)]
    #P_I_list = variables[(K+2):(2*K+3)]
    print('thresholds:', mode)
    if way == 'lamda':
        if lamda is None:
            lamda = 0.02

            print('lamda = ', 0.02)
        else:
            print('lamda = ', lamda)
    else:
        lamda = lambda_
        print("lambda = ", lamda)
    if need_print:
        for k in range(K+1):
            print("P_A_", k, " = ", P_A_list[k], \
                    "                 P_I_", k, " = ", P_I_list[k])

    """ variables = res_value.x
    lamda_value = variables[0]
    P_A_list_value = variables[1:(K+2)]
    P_I_list_value = variables[(K+2):(2*K+3)]
    if need_print:
        print("lambda = ", lamda_value)
        for k in range(K+1):
            print("P_A_", k, " = ", P_A_list_value[k], \
                  "                 P_I_", k, " = ", P_I_list_value[k]) """
    return lamda, P_A_list, P_I_list



def get_P_AI_K_dict(K_list, P_A_dict, P_I_dict, nX=None, nY=None, nZ=None, \
                    nR=None, nT=None, nS=None, mode='value'):
    P_AI_K_dict = {}
    for K in K_list:
        P_AI_K_dict[K] = []
        for k in range(K, 0, -1):
            P_AI_K_dict[K].append((sum(P_A_dict[K][k:]), sum(P_I_dict[K][k:])))
    return P_AI_K_dict




def interpolate_plot(K_list, P_AI_K_dict, lamda_dict, need_plot=True, nX=None, nY=None, \
                     nZ=None, nR=None, nT=None, nS=None, mode='value', contrast_name=''):
    best_P_I = {}
    best_P_A = {}
    P_Ah = {}
    P_Ih = {}
    for K in K_list:
        #P_Ih[K] = np.linspace(0, 1, 1000)
        P_A_list = [P_AI_K_dict[K][k][0] for k in range(len(P_AI_K_dict[K]))]
        P_I_list = [P_AI_K_dict[K][k][1] for k in range(len(P_AI_K_dict[K]))]
        #P_I_list = P_AI_K_dict[K][:][1]
        print('P_A:')
        print(P_A_list)
        print('P_I:')
        print(P_I_list)
        lamda = lamda_dict[K]
        #spl = UnivariateSpline(P_I_list, P_A_list)
        #P_Ah[K] = spl(P_Ih[K])
        E_opt, R_opt, P_Ih[K], P_Ah[K] = get_curve(P_A_list, P_I_list, n_points=1000)

        Kappah = [get_Kappa(P_Ih[K][i], P_Ah[K][i], lamda) for i in range(P_Ih[K].shape[0])]
        best_index, best_Kappa = max(enumerate(Kappah), key=lambda pair: pair[1])
        best_P_I[K] = P_Ih[K][best_index]
        best_P_A[K] = P_Ah[K][best_index]
        if need_plot:
            fig = plt.figure(figsize=(5, 5))
            plt.plot(P_Ih[K], P_Ah[K], label="Байес")
            plt.plot(P_Ih[K], Kappah, label="Каппа-функция")
            plt.scatter(best_P_I[K], best_P_A[K], label='Рабочая точка')
            plt.scatter(P_I_list, P_A_list, label='Исходные точки')
            plt.title(f"ROC-кривая для K = {K}, contrast = {contrast_name}")
            plt.xlabel('P_I')
            plt.ylabel('P_A')
            plt.legend()
            plt.show()
    return best_P_I, best_P_A, P_Ih, P_Ah




def get_Kappa(P_I, P_A, lamda, nX=None, nY=None, nZ=None, \
              nR=None, nT=None, nS=None):
    p0 = lamda * P_A + (1 - lamda) * (1 - P_I)
    tau = lamda * P_A + (1 - lamda) * P_I
    pC = lamda * tau + (1 - lamda) * (1 - tau)
    kappa = (p0 - pC) / (1 - pC)
    return kappa




def get_best_thresholds(K_list, t_values, nX=None, nY=None, nZ=None, \
                        nR=None, nT=None, nS=None, mode='value', way='lamda', contrast_name = ''):
    final_thresholds = {}
    #final_thresholds['value'] = {}
    #final_thresholds['numb'] = {}
    
    dict_thresholds = get_thresholds_dict(K_list, t_values, nX, nY, nZ, nR, nT, nS, mode)
    
    t_values_voxel_groups = get_t_groups(K_list, dict_thresholds, t_values, nX, nY, nZ, nR, nT, nS, mode)

    P_Ak_dict = {}
    P_Ik_dict = {}
    lamda_dict = {}
    #P_A_dict['value'] = {}
    #P_A_dict['numb'] = {}
    #P_I_dict['value'] = {}
    #P_I_dict['numb'] = {}
    #lamda_dict['value'] = {}
    #lamda_dict['numb'] = {}
    
    for K in K_list:
        #P_A_k
        lamda, P_A_list, P_I_list = print_get_solution(K, t_values_voxel_groups[K], False, nX, nY, nZ, nR, nT, nS, mode, way)
        P_Ak_dict[K] = copy.deepcopy(P_A_list)
        #P_A_dict['value'][K] = copy.deepcopy(P_A_list_value)
        #P_A_dict['numb'][K] = copy.deepcopy(P_A_list_numb)
        P_Ik_dict[K] = copy.deepcopy(P_I_list)
        #P_I_dict['value'][K] = copy.deepcopy(P_I_list_value)
        #P_I_dict['numb'][K] = copy.deepcopy(P_I_list_numb)
        lamda_dict[K] = lamda
        #lamda_dict['value'][K] = lamda_value
        #lamda_dict['numb'][K] = lamda_numb
        print(f'K = {K}')
        print('thresholds:', mode)
        print('lamda', lamda)
        print('P_A_k', P_A_list)
        print('P_I_k', P_I_list)
        #print('lamda_numb', lamda_numb)
        #print('P_A_numb', P_A_list_numb)
        #print('P_I_numb', P_I_list_numb)
        #print('lamda_numb', lamda_value)
        #print('P_A_value', P_A_list_value)
        #print('P_I_value', P_I_list_value)
    P_AI_K_dict = {}
    P_AI_K_dict = get_P_AI_K_dict(K_list, P_Ak_dict, P_Ik_dict, nX, nY, nZ, nR, nT, nS, mode)
    print('P_AI_K_dict')
    print(P_AI_K_dict)
    #P_AI_K_dict['value'] = get_P_AI_K_dict(K_list, P_A_dict['value'], P_I_dict['value'], nX, nY, nZ, nR, nT, nS)
    #P_AI_K_dict['numb'] = get_P_AI_K_dict(K_list, P_A_dict['numb'], P_I_dict['numb'], nX, nY, nZ, nR, nT, nS)
    
    best_P_I = {}
    best_P_A = {}
    #best_P_I['value'] = {}
    #best_P_I['numb'] = {}
    #best_P_A['value'] = {}
    #best_P_A['numb'] = {}

    P_Ih = {}
    P_Ah = {}
    #P_Ih['value'] = {}
    #P_Ih['numb'] = {}
    #P_Ah['value'] = {}
    #P_Ah['numb'] = {}
    

    best_P_I, best_P_A, P_Ih, P_Ah = \
    interpolate_plot(K_list, P_AI_K_dict, lamda_dict, True, nX, nY, nZ, nR, nT, nS, mode, contrast_name=contrast_name)
    #best_P_I, best_P_A['value'], P_Ih['value'], P_Ah['value'] = \
    #interpolate_plot(K_list, P_AI_K_dict['value'], lamda_dict['value'], True, nX, nY, nZ, nR, nT, nS)
    #best_P_I['numb'], best_P_A['numb'], P_Ih['numb'], P_Ah['numb'] = \
    #interpolate_plot(K_list, P_AI_K_dict['numb'], lamda_dict['numb'], True, nX, nY, nZ, nR, nT, nS)
    P_A_dict = {}
    P_I_dict = {}
    for K in K_list:
        P_A_dict[K] = np.array([P_AI_K_dict[K][k][0] for k in range(len(P_AI_K_dict[K]))])
        P_I_dict[K] = np.array([P_AI_K_dict[K][k][1] for k in range(len(P_AI_K_dict[K]))])

    for K in K_list:
        count = 0
        #count1 = 0
        pare_next = (P_I_dict[K][1], P_A_dict[K][1])
        pare_prev = (P_I_dict[K][0], P_A_dict[K][0])
        n_pare = 1
        #count2 = 0
        for i in range(P_I_dict[K].shape[0]):
            #P_I = {}
            #P_A = {}
            P_I = P_Ih[K][i]
            P_A = P_Ah[K][i]
            #P_I['value'] = P_Ih['value'][K][i]
            #P_A['value'] = P_Ah['value'][K][i]
            #P_I['numb'] = P_Ih['numb'][K][i]
            #P_A['numb'] = P_Ah['numb'][K][i]
            if (P_A > best_P_A[K]) and (P_I > best_P_I[K]) and (count == 0):
                pare_next = (P_I_dict[K][i], P_A_dict[K][i])
                pare_prev = (P_I_dict[K][i-1], P_A_dict[K][i-1])
                n_pare = i
                count = 1
            #if (P_A['numb'] > best_P_A['numb'][K]) and (P_I['numb'] > best_P_I['numb'][K]) and (count2 == 0):
            #    pare_next['numb'] = (P_Ih['numb'][K][i], P_Ah['numb'][K][i])
            #    pare_prev['numb'] = (P_Ih['numb'][K][i-1], P_Ah['numb'][K][i-1])
            #    n_pare['numb'] = i
            #    count2 = 1
            #if count1 + count2 == 2:
            #    break
            if count == 1:
                break
        threshold_prev = {}
        threshold_next = {}
        koef = {}
        #print([n_pare['numb']-1])
        #print(dict_thresholds['numb'][K][n_pare['numb']-1])
        threshold_prev = dict_thresholds[K][n_pare-1]
        threshold_next = dict_thresholds[K][n_pare]
        #threshold_prev['numb'] = dict_thresholds['numb'][K][n_pare['numb']-1]
        #threshold_next['numb'] = dict_thresholds['numb'][K][n_pare['numb']]
        #threshold_prev['value'] = dict_thresholds['value'][K][n_pare['value']-1]
        #threshold_next['value'] = dict_thresholds['value'][K][n_pare['value']]
        koef = 0.5 * ((best_P_A[K] - pare_prev[1])/(pare_next[1] - pare_prev[1]) \
                      + (best_P_I[K] - pare_prev[0])/(pare_next[0] - pare_prev[0]))
        #koef['numb'] = 0.5 * ((best_P_A['numb'][K] - pare_prev['numb'][1])/(pare_next['numb'][1] - pare_prev['numb'][1]) \
        #              + (best_P_I['numb'][K] - pare_prev['numb'][0])/(pare_next['numb'][0] - pare_prev['numb'][0]))
        #koef['value'] = 0.5 * ((best_P_A['value'][K] - pare_prev['value'][1])/(pare_next['value'][1] - pare_prev['value'][1]) \
        #              + (best_P_I['value'][K] - pare_prev['value'][0])/(pare_next['value'][0] - pare_prev['value'][0]))
        final_thresholds[K] = threshold_prev + (threshold_next - threshold_prev) * koef
        #final_thresholds['numb'][K] = threshold_prev['numb'] + (threshold_next['numb'] - threshold_prev['numb']) * koef['numb']
        #final_thresholds['value'][K] = threshold_prev['value'] + (threshold_next['value'] - threshold_prev['value']) * koef['value']
    return final_thresholds, P_Ak_dict, P_Ik_dict, lamda_dict



def get_reproducibility(K_list, final_thresholds, t_values, \
                        nX=None, nY=None, nZ=None, nR=None, nT=None, nS=None, mode='value'):
    #print(t_values.shape)
    voxel_status_dict = {}
    for K in K_list:
        voxel_status_dict[K] = np.zeros((t_values.shape[0]))
        #print(voxel_status_dict.keys())
        #print(voxel_status_dict[K])
        for v in range(t_values.shape[0]):
            voxel_status_dict[K][v] = sum([(t_val > final_thresholds[K]) \
                                           for t_val in t_values[v]]) / t_values[v].shape[0]
    return voxel_status_dict


def iter_triple(x_values, y_values, z_values):
    answer = []
    for x in x_values:
        for y in y_values:
            for z in z_values:
                answer.append((x, y, z))
    return answer

def check_neighbors(data_array, x, y, z, nX=None, nY=None, nZ=None, nR=None, nT=None, nS=None, strong_threshold=0.9, medium_threshold=0.7):
    if x == 0:
        x_values = [0, 1]
    elif x == nX-1:
        x_values = [nX-2, nX-1]
    else:
        x_values = [x-1, x, x+1]

    if y == 0:
        y_values = [0, 1]
    elif y == nY-1:
        y_values = [nY-2, nY-1]
    else:
        y_values = [y-1, y, y+1]
    
    if z == 0:
        z_values = [0, 1]
    elif z == nZ-1:
        z_values = [nZ-2, nZ-1]
    else:
        z_values = [z-1, z, z+1]

    indexes_array = iter_triple(x_values, y_values, z_values)
    indexes_array.remove((x, y, z))

    if data_array[x, y, z] >= strong_threshold:
        return 'high'
    elif any(data_array[index] >= strong_threshold for index in indexes_array) and data_array[x,y,z] >= medium_threshold:
        return 'medium'
    else:
        return 'low'

'''
def check_neighbours(data_array, x, y, z, nX=None, nY=None, nZ=None, nR=None, nT=None, nS=None):
    nX, nY, nZ = data_array.shape
    indexes_array = []
    if ((x == 0) and (y == 0) and (z == 0)):
        indexes_array = [
            [0, 0, 1], [1, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]
        ]
    elif ((x == nX - 1) and (y == nY - 1) and (z == nZ - 1)):
        indexes_array = [
            [nX-1, nY-1, nZ-2], [nX-2, nY-1, nZ-1], [nX-1, nY-2, nZ-1], [nX-2, nY-2, nZ-1], [nX-2, nY-1, nZ-2], [nX-1, nY-2, nZ-2], [nX-2, nY-2, nZ-2]
        ]

    elif ((x == 0) and (y == 0)):
        new_indexes = iter_triple([0, 1], [0, 1], [z-1, z, z+1])
        #indexes_array.extend([[[a, b, z-1] for a in [0, 1]] for b in [0, 1]])
        #indexes_array.extend([[[a, b, z+1] for a in [0, 1]] for b in [0, 1]])
        #indexes_array.extend([[[a, b, z] for a in [0, 1]] for b in [0, 1]])
        indexes_array.extend(new_indexes)
        indexes_array.remove([0, 0, z])
    elif ((x == 0) and (z == 0)):
        new_indexes = iter_triple([0, 1], [0, 1], [z-1, z, z+1])
        #indexes_array.extend([[[a, y-1, c] for a in [0, 1]] for c in [0, 1]])
        #indexes_array.extend([[[a, y+1, c] for a in [0, 1]] for c in [0, 1]])
        #indexes_array.extend([[[a, y, c] for a in [0, 1]] for c in [0, 1]])
        indexes_array.extend(new_indexes)
        indexes_array.remove([0, y, 0])
    elif ((y == 0) and (z == 0)):
        new_indexes = iter_triple([x-1, x, x+1], [0, 1], [0, 1])
        #indexes_array.extend([[[x-1, b, c] for b in [0, 1]] for c in [0, 1]])
        #indexes_array.extend([[[x+1, b, c] for b in [0, 1]] for c in [0, 1]])
        #indexes_array.extend([[[x, b, c] for b in [0, 1]] for c in [0, 1]])
        indexes_array.extend(new_indexes)
        indexes_array.remove([x, 0, 0])
    elif ((x == nX - 1) and (y == nY - 1)):
        new_indexes = iter_triple([nX-2, nX-1], [nY-2, nY-1], [z-1, z, z+1])
        #indexes_array.extend([[[a, b, z-1] for a in [nX-2, nX-1]] for b in [nY-2, nY-1]])
        #indexes_array.extend([[[a, b, z+1] for a in [nX-2, nX-1]] for b in [nY-2, nY-1]])
        #indexes_array.extend([[[a, b, z] for a in [nX-2, nX-1]] for b in [nY-2, nY-1]])
        indexes_array.extend(new_indexes)
        indexes_array.remove([nX-1, nX-1, z])
    elif ((x == nX - 1) and (z == nZ - 1)):
        new_indexes = iter_triple([nX-2, nX-1], [y-1, y, y+1], [nZ-2, nZ-1])
        #indexes_array.extend([[[a, y-1, c] for a in [nX-2, nX-1]] for c in [nZ-2, nZ-1]])
        #indexes_array.extend([[[a, y+1, c] for a in [nX-2, nX-1]] for c in [nZ-2, nZ-1]])
        #indexes_array.extend([[[a, y, c] for a in [nX-2, nX-1]] for c in [nZ-2, nZ-1]])
        indexes_array.extend(new_indexes)
        indexes_array.remove([nX-1, y, nZ-1])
    elif ((y == nY - 1) and (z == nZ - 1)):
        new_indexes = iter_triple([x-1, x, x+1], [nY-2, nY-1], [nZ-2, nZ-1])
        #indexes_array.extend([[[x-1, b, c] for b in [nY-2, nY-1]] for c in [nZ-2, nZ-1]])
        #indexes_array.extend([[[x+1, b, c] for b in [nY-2, nY-1]] for c in [nZ-2, nZ-1]])
        #indexes_array.extend([[[x, b, c] for b in [nY-2, nY-1]] for c in [nZ-2, nZ-1]])
        indexes_array.extend(new_indexes)
        indexes_array.remove([x, nY-1, nZ-1])
    elif ((x == 0)):
        new_indexes = iter_triple([0, 1], [nY-2, nY-1], [nZ-2, nZ-1])
        #indexes_array.extend([[[[a, b, c] for a in [0, 1]] for b in [y-1, y, y+1]] for c in [z-1, z, z+1]])
        indexes_array.extend(new_indexes)
        indexes_array.remove([0, y, z])
    elif ((y == 0)):
        indexes_array.extend([[[[a, b, c] for a in [x-1, x, x+1]] for b in [0, 1]] for c in [z-1, z, z+1]])
        indexes_array.remove([x, 0, z])
    elif ((z == 0)):
        indexes_array.extend([[[[a, b, c] for a in [x-1, x, x+1]] for b in [y-1, y, y+1]] for c in [0, 1]])
        indexes_array.remove([x, y, 0])
    else:
        indexes_array.extend([[[[a, b, c] for a in [x-1, x, x+1]] for b in [y-1, y, y+1]] for c in [z-1, z, z+1]])
        indexes_array.remove([x, y, z])

    if data_array[x, y, z] >= 0.9:
        return 'high'
    elif any(data_array[index] >= 0.9 for index in indexes_array) and data_array[x,y,z] >= 0.7:
        return 'medium'
    else:
        return 'low'
    '''

from scipy.special import logsumexp

def EM_optimization(r, K, max_iter=100, tol=1e-6, nX=None, nY=None, nZ=None, nT=None, nR=None, nS=None, mode='lamda', lamda=0.02, n_repeats=100):
    #lambda_dict = {}
    #P_A_dict = {}
    #P_I_dict = {}
    N, _ = r.shape
    M = np.sum(r[0])

    #print('Число t-значений по группам: \n', r)
    #Инициализация
    if mode == 'lamda':
        lambda_ = lamda
    else:
        lambda_ = 0.5

    start_lambda = []
    start_P_A = []
    start_P_I = []
    solutions_lambda = []
    solutions_P_A = []
    solutions_P_I = []

    for n_rep in range(n_repeats):
        delta_P_A = sps.uniform.rvs(size=K+1, loc = -1/(K+2), scale = 2/(K+2))
        delta_P_I = sps.uniform.rvs(size=K+1, loc = -1/(K+2), scale = 2/(K+2))
        #delta_P_A = np.array([1/(2*K+2) if i % 2 == 0 else -1/(2*K+2) for i in range(K+1)])
        #delta_P_I = np.array([1/(2*K+2) if i % 2 == 1 else -1/(2*K+2) for i in range(K+1)])
        #rand_n1 = np.random.randint(0, K+1)
        #rand_n2 = np.random.randint(0, K+1)
        #delta_P_A[rand_n1] = - (np.sum(delta_P_A) - delta_P_A[rand_n1])
        #delta_P_I[rand_n2] = - (np.sum(delta_P_I) - delta_P_I[rand_n2])
        #sum_delta_P_A = np.sum(delta_P_A)
        #sum_delta_P_I = np.sum(delta_P_I)
        #for i in range(len(delta_P_A)):
        #    delta_P_A[i] -= sum_delta_P_A/len(delta_P_A)
        #    delta_P_I[i] -= sum_delta_P_I/len(delta_P_I)
        P_A = np.ones(K+1) / (K+1) + delta_P_A
        P_I = np.ones(K+1) / (K+1) + delta_P_I
        sum_P_A = np.sum(P_A)
        sum_P_I = np.sum(P_I)
        for i in range(len(P_A)):
            P_A[i] /= sum_P_A
            P_I[i] /= sum_P_I
        if mode != 'lamda':
            lambda_ = sps.uniform.rvs(size=1, loc=0, scale=1)[0]
        print('start_values rep', n_rep, ':')
        print('P_A_start:')
        print(P_A)
        start_P_A.append(P_A)
        print('P_I_start:')
        print(P_I)
        start_P_I.append(P_I)
        print('lambda_start:')
        print(lambda_)
        start_lambda.append(lambda_)
        #print('Начали оптимизацию')
        #print(r)
        n_iters = 0
        for i in range(max_iter):
            #print('Итерация', i)
            #print('lambda:', lambda_)
            #print('P_A:\n', P_A)
            #print('P_I:\n', P_I)
            # E-шаг: вычисление gamma_v
            log_P_A = np.log(P_A + 1e-10)
            log_P_I = np.log(P_I + 1e-10)
            log_lambda = np.log(lambda_)
            log_1m_lambda = np.log(1 - lambda_)
            
            log_gamma_A = r @ log_P_A.T + log_lambda
            #print('log_gamma_A', log_gamma_A.shape)
            log_gamma_I = r @ log_P_I.T + log_1m_lambda
            #print('log_gamma_A', log_gamma_I.shape)
            
            gamma = np.exp(log_gamma_A - logsumexp(np.vstack([log_gamma_A, log_gamma_I]), axis=0))
            #if mode == 'lamda':
            #    sum_gamma = np.sum(gamma)
            #    for v in range(gamma.shape[0]):
            #        gamma[v] = gamma[v] / sum_gamma
            #print('gamma:\n', gamma)

            #for v in range(N):
            #    print(f'gamma[{v}]')
            #    print(f'{gamma[v]} = ({np.exp(log_gamma_A[v])}) / ({np.exp(log_gamma_A[v])} + {np.exp(log_gamma_I[v])})')

            # M-шаг: обновление параметров
            if mode == 'lamda':
                lambda_new = lambda_
            else:
                lambda_new = np.mean(gamma)
            #print('lambda_new:', lambda_new)
            sum_gamma = np.sum(gamma)
            sum_1m_gamma = N - sum_gamma
            
            #P_A_new = np.sum(gamma[:, None] * r, axis=0) / (M * sum_gamma)
            #P_A_new = np.zeros(K+1)
            #for k in range(K+1):
            #    P_A_new[k] = np.sum([gamma[v] * r[v, k] for v in range(N)])
            P_A_new = gamma.T @ r / (M * sum_gamma)
            #print('P_A_new:\n', P_A_new)

            #for k in range(K+1):
            #    print(f'P_A_{k}')
            #    print(f'{P_A_new[k]} = ({(gamma.T @ r)[k]}) / ({M} * {sum_gamma})')

            #P_I_new = np.sum((1 - gamma)[:, None] * r, axis=0) / (M * sum_1m_gamma)
            P_I_new = (1 - gamma).T @ r/ (M * sum_1m_gamma)
            #print('P_I_new:\n', P_I_new)

            #for k in range(K+1):
            #    print(f'P_I_{k}')
            #    print(f'{P_I_new[k]} = ({((1-gamma).T @ r)[k]}) / ({M} * {sum_1m_gamma})')
            
            # Проверка сходимости
            if ((np.abs(lambda_new - lambda_)) < tol or mode == 'lamda') and \
                np.all(np.abs(P_A_new - P_A) < tol) and \
                np.all(np.abs(P_I_new - P_I) < tol) or i == max_iter-1:
                n_iters = i + 1
                solutions_lambda.append(lambda_)
                solutions_P_A.append(P_A.copy())
                solutions_P_I.append(P_I.copy())
                break
            #print(((np.abs(lambda_new - lambda_)) < tol or mode == 'lamda'), np.all(np.abs(P_A_new - P_A) < tol), np.all(np.abs(P_I_new - P_I) < tol))
            #print(np.abs(P_A_new - P_A),'\n', np.abs(P_I_new - P_I))
                
            
            #if mode != 'lamda':
            lambda_, P_A, P_I = lambda_new, P_A_new, P_I_new
            #else:
            #    P_A, P_I = P_A_new, P_I_new
            #print('lambda:', lambda_)
            #print('P_A:\n', P_A)
            #print('P_I:\n', P_I)
        print('P_A:')
        print(P_A)
        print('P_I:')
        print(P_I)
        print('lambda:')
        print(lambda_)

    best_P_A_list = solutions_P_A[0]
    best_P_I_list = solutions_P_I[0]
    best_lambda = solutions_lambda[0]
    best_f = f_obj(best_lambda, best_P_A_list, best_P_I_list, r)
    f_obj_values = []
    start_f_values = []
    for i in range(n_repeats):
        P_A_list = solutions_P_A[i]
        P_I_list = solutions_P_I[i]
        lambda_ = solutions_lambda[i]
        curr_f = f_obj(lambda_, P_A_list, P_I_list, r)
        start_f = f_obj(start_lambda[i], start_P_A[i], start_P_I[i], r)
        start_f_values.append(start_f)
        f_obj_values.append(curr_f)
        if curr_f > best_f:
            best_f = curr_f
            best_P_A_list = P_A_list.copy()
            best_P_I_list = P_I_list.copy()
            best_lambda = lambda_

    lambda_ = best_lambda
    P_A = best_P_A_list
    P_I = best_P_I_list

    print('Остановка ЕМ')
    for i in range(len(f_obj_values)):
        print(f'{i}-й запуск')
        print('start:', start_f_values[i])
        print('final:', f_obj_values[i])
    #print(f_obj_values)
    if n_iters != 0:
        print('Остановились на итерации', n_iters)
    else:
        print('Закончили по количеству итераций')
    print('mode:', mode)
    print('lambda:', lambda_)
    #print('lambda_new:', lambda_new)
    print('P_A:\n', P_A)
    #print('P_A_new:\n', P_A_new)
    print('P_I:\n', P_I)
    #print('P_I_new:\n', P_I_new)
    print(sum(P_A), sum(P_I))


    return lambda_, P_A, P_I

def exponential_roc_model(P_I, E, R):
    """Экспоненциальная модель ROC-кривой."""
    return R * (P_I**(1/E)) + (1 - R)*(1 - (1 - P_I)**E)

def objective_function(E, R, P_I_observed, P_A_observed):
    """Функция для минимизации ошибки аппроксимации."""
    P_A_predicted = [exponential_roc_model(P_I, E, R) for P_I in P_I_observed]
    error = np.sum([(P_A_predicted[i] - P_A_observed[i])**2 for i in range(len(P_I_observed))])
    return error

def get_curve(P_A_observed, P_I_observed, initial_E=1.0, initial_R=0, need_plot=False, n_points=100):
    # Пример данных (пары P_I и P_A)
    #P_I_observed = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    #P_A_observed = np.array([0.0, 0.15, 0.3, 0.45, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0])

    # Начальное предположение для theta
    #initial_thet
    matrix = [[1, 0], [0, 1]]
    lb = [0.1, 0]
    ub = [10, 1]
    const = LinearConstraint(matrix, lb, ub)

    # Оптимизация параметра theta
    result = scipy.optimize.minimize(lambda x: objective_function(x[0], x[1], P_I_observed, P_A_observed), [0.1, 0.5], constraints=const)
    #, args=(P_I_observed, P_A_observed)
    E_optimized = result.x[0]
    R_optimized = result.x[1]
    
    print('Лучшее значение E:', E_optimized)
    print('Лучшее значение R:', R_optimized)
    print('Ошибка:', result.fun)
    print('Ошибка при начальных E M:', objective_function(initial_E, initial_R, P_I_observed, P_A_observed))

    # Генерация гладкой ROC-кривой
    P_I_smooth = np.linspace(0, 1, 100)
    P_A_smooth = exponential_roc_model(P_I_smooth, E_optimized, R_optimized)

    if need_plot:
        # Визуализация
        plt.figure(figsize=(8, 6))
        plt.plot(P_I_observed, P_A_observed, 'ro', label='Наблюдаемые точки')
        plt.plot(P_I_smooth, P_A_smooth, 'b-', label=f'Экспоненциальная модель (E={E_optimized:.2f}, R={R_optimized:.2f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Случайное угадывание')
        plt.xlabel('Вероятность ложного срабатывания (P_I)')
        plt.ylabel('Чувствительность (P_A)')
        plt.title('ROC-кривая: экспоненциальная модель')
        plt.legend()
        plt.grid(True)
        plt.show()
    return E_optimized, R_optimized, P_I_smooth, P_A_smooth

def get_map_array(K_list, voxel_status_dict, \
                  nX=None, nY=None, nZ=None, nR=None, nT=None, nS=None, mode='value', strong_threshold=0.9, normal_threshold=0.7):
    nV = nX * nY * nZ
    reproducibility_array = {}
    coord_voxel_array = {}
    for K in K_list:
        coord_voxel_array[K] = np.zeros((nX, nY, nZ))
        reproducibility_array[K] = [[[0 for _ in range(nZ)] \
                                     for _ in range(nY)] for _ in range(nX)]
        for v in range(voxel_status_dict[K].shape[0]):
            coord_voxel_array[K][voxel_coord(v,nX,nY,nZ)] = voxel_status_dict[K][v]
        for v in range(voxel_status_dict[K].shape[0]):
            if coord_voxel_array[K][voxel_coord(v, nX, nY, nZ)] >= strong_threshold:
                coord_voxel_array[K][voxel_coord(v, nX, nY, nZ)] = 1
        for v in range(voxel_status_dict[K].shape[0]):
            x, y, z = voxel_coord(v, nX, nY, nZ)
            reproducibility_array[K][x][y][z] = check_neighbors(coord_voxel_array[K], \
                                                                              x, y, z,
                                                                             nX, nY, nZ, nR, nT, nS, strong_threshold, normal_threshold)
    return reproducibility_array, coord_voxel_array





def plot_map(voxel_status_array, t_values, x_list=None, y_list=None, z_list=None, \
             nX=None, nY=None, nZ=None, nR=None, nT=None, nS=None, array_to_show=None, contrast_name='', mode='str'):
    if contrast_name != '':
        contrast_line = '_' + contrast_name+' contrast'
    plot_slices = []
    plot_names = []
    if mode == 'str':
        new_voxel_array = np.zeros((len(voxel_status_array), len(voxel_status_array[0]), len(voxel_status_array[0][0])))
        for x in range(new_voxel_array.shape[0]):
            for y in range(new_voxel_array.shape[1]):
                for z in range(new_voxel_array.shape[2]):
                    if voxel_status_array[x][y][z] == 'high':
                        new_voxel_array[x,y,z] = 1
                    elif voxel_status_array[x][y][z] == 'medium':
                        new_voxel_array[x,y,z] = 1
                    else:
                        voxel_status_array[x][y][z] = 0

        for x in range(new_voxel_array.shape[0]):
            for y in range(new_voxel_array.shape[1]):
                for z in range(new_voxel_array.shape[2]):
                    new_voxel_array[x, y, z] *= np.mean(t_values[coord_voxel(x, y, z, nX, nY, nZ)])
    else:
        new_voxel_array = array_to_show.copy()

    axis_slice = []

    if x_list != None:
        for x in x_list:
            plot_names.append(f'Срез по x={x}')
            axis_slice.append(('y', 'z'))
            plot_slices.append(new_voxel_array[x, :, :])
    if y_list != None:
        for y in y_list:
            plot_names.append(f'Срез по y={y}')
            axis_slice.append(('x', 'z'))
            plot_slices.append(new_voxel_array[:, y, :])
    if z_list != None:
        for z in z_list:
            plot_names.append(f'Срез по z={z}')
            axis_slice.append(('x', 'y'))
            plot_slices.append(new_voxel_array[:, :, z])


    for i, plot_slice in enumerate(plot_slices):
        fig = plt.figure(figsize=(5, 5))
        #print(plot_slice)
        plt.imshow(plot_slice, cmap='Greys_r')
        #plt.imshow(plot_slice, cmap='gray', origin='lower')
        plt.title(plot_names[i]+contrast_line)
        plt.xlabel(axis_slice[i][0] + ' axis')
        plt.ylabel(axis_slice[i][1] + ' axis')
        #plt.ylabel('Second axis')
        plt.colorbar(label='Значения t-статистики')
        plt.clim(0, 1)
        plt.axis('off')
        plt.show()

    while True:
        print(contrast_name)
        slice_str = input(f'Введите срез формата x/y/z = n или end, contrast = ' + contrast_name + ': ')
        if 'end' in slice_str:
            break
        
        if slice_str.count('=') > 1:
            print('too many "=" in str')
            continue

        if '=' in slice_str:
            slice_str = slice_str.replace(' ', '')
            var, n = slice_str.split('=')
            if var in ['x', 'y', 'z']:
                if n.isdigit():
                    if var == 'x':
                        x = int(n)
                        name = f'Срез по x = {x}'
                        slice = new_voxel_array[x, :, :]
                        axis = ['y', 'z']
                    elif var == 'y':
                        y = int(n)
                        name = f'Срез по y = {y}'
                        slice = new_voxel_array[:, y, :]
                        axis = ['x', 'z']
                    else:
                        z = int(n)
                        name = f'Срез по z = {z}'
                        slice = new_voxel_array[:, :, z]
                        axis = ['x', 'y']
                    fig = plt.figure(figsize=(5, 5))
                    plt.imshow(slice, cmap='Greys_r')
                    plt.title(name)
                    plt.xlabel(axis[0] + ' axis')
                    plt.ylabel(axis[1] + ' axis')
                    plt.colorbar(label='t-values')
                    plt.clim(0, 1)
                    plt.axis('off')
                    plt.show()
                else:
                    print('Invalid number of slice:', n, 'is not a number')
                    continue
            else:
                print('Invalid name of axe:', var, 'not in [x, y, z]')

        else:
            print('no "=" in str')
            continue

    return new_voxel_array
        



def computing(X_matrixes, patient_data_dict, contrasts, K_list, X2, voxels_list=None, \
                             x_list=None, y_list=None, z_list=None, nX=None, \
            nY=None, nZ=None, nR=None, nT=None, nS=None, return_dict=None, max_iter=100, runs=None) -> Tuple[Dict, Dict]:
    nV = nX * nY * nZ
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=IntegrationWarning)
    warnings.filterwarnings("ignore", message='.*cgi.*', category=DeprecationWarning)
    warnings.filterwarnings("ignore", message=".*'cgi' is deprecated.*", category=DeprecationWarning)
    try:
        print(f"Process {os.getpid()} started with {len(voxels_list)} voxels", flush=True)
        #print('Computing', voxels_list)
        if voxels_list is None:
            voxels_list = []
            for x in range(nX):
                for y in range(nY):
                    for z in range(nZ):
                        voxels_list.append(coord_voxel(x, y, z, nX, nY, nZ))
        t_values = {}
        T_values = {}
        for contrast_name in contrasts.keys():
            t_values[contrast_name] = {}
            T_values[contrast_name] = {}
        t0 = time.time()
        for i, v in enumerate(voxels_list):
            if i % 100 == 1:
                print(os.getpid(), i, '/', len(voxels_list))
            Y_list = []
            t1 = time.time()
            x, y, z = voxel_coord(v, nX, nY, nZ)
            #print(z, len(X_matrixes))
            X_list = X_matrixes[z]
            #Y_list = get_Y(patient_data_path, voxel = (x, y, z))
            for run in runs:
                Y_list.append(np.array([scan[x,y,z] for scan in patient_data_dict[run]]))
            ta = time.time()
            #print('Загрузили за', ta-t1)
            #mean_beta, disp_beta = run_regression(patient_data_path, \
            #                  patient_logs_path, (x, y, z), X2)
            #print('X_list', X_list.shape)
            #for i in range(len(X_list)):
            #    print('Эксперимент', i)
            #    print_matrix(X_list[i], 'X_list')
            #    print_vector(Y_list[i], 'Y_list')
            mean_beta, disp_beta, mu, Omega = EM_new(X_list, Y_list, X2, nX, nY, nZ, nR, nT, nS, max_iter=max_iter, voxel=i)
            tb = time.time()
            #print('EM за', tb-ta)
            #print(mean_beta.shape, disp_beta.shape)
            for contrast_name, contrast in contrasts.items():
                t_value_list = get_t_value(mean_beta, disp_beta, contrast, nX, nY, nZ, nR, nT, nS)
                T_value = get_T_value(mu, Omega, contrast, nX, nY, nZ, nR, nT, nS)
                tc = time.time()
                #print('Получили t значения за', tc-tb)
                #print(t_value_list, '\n', T_value)
                t_values[contrast_name][v] = np.array(t_value_list)
                T_values[contrast_name][v] = T_value
                del t_value_list
                del T_value
            del Y_list
            del mean_beta
            del disp_beta
                #t2 = time.time()
                    #print(f'Затрачено времени t={t2-t1}')
        t1 = time.time()
        print(os.getpid() ,len(voxels_list), 'вокселей за', t1-t0, 'секунд')
        '''
        for v in voxels_list:
            return_dict['t'][v] = t_values[v].copy
            return_dict['T'][v] = T_values[v].copy
        '''
        #print(is_serializable(t_values), is_serializable(T_values))
        #print(total_size(t_values), total_size(T_values))
        return t_values, T_values
        #with tempfile.NamedTemporaryFile(delete=False, suffix='.npz', dir='.', prefix='resulf_', mode='wb') as f:
        #    np.savez(f, t_values=t_values, T_values=T_values)
        #    return f.name
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        with open(f"errors_voxels_{os.getpid()}.log", "w") as log_file:
            log_file.write("Exception in computing:\n")
            log_file.write(traceback.format_exc())
        raise e





def multiprocessed(patient_data_path, patient_logs_path, contrast, K_list, X2, \
                             x_list=None, y_list=None, z_list=None, nX=None, nY=None, nZ=None, nR=None, nT=None, nS=None, \
                        procs=None, calcs=None):
    nV = nX * nY * nZ
    print(procs, len(calcs))
    processes = []
    t_values_list = [{} for _ in range(procs)]
    T_values_list = [{} for _ in range(procs)]
    t_values = np.zeros((nV, nR))
    T_values = np.zeros(nV)
    queues = [0 for _ in range(procs)]
    #manager = multiprocessing.Manager()
    #return_dict = manager.dict()
    #return_dict['t'] = {}
    #return_dict['T'] = {}
    '''
    # Запуск мультипроцессинга
    for proc in range(procs):
        #print('Go', proc)
        #print('1a', proc)
        p = multiprocessing.Process(target=computing, args=(patient_data_path, \
                             patient_logs_path, contrast, K_list, X2, calcs[proc], \
                             x_list, y_list, z_list, nX, nY, nZ, nR, return_dict))
        print(p)
        #print('1b', proc)
        processes.append(p)
        #print('1c', proc)
        p.start()
        #print('1d', proc)
    #Завершение мультипроцессинга    
    for p in processes:
        #print('Finish', proc)
        p.join()
        #print('1f', proc)
    #Соеднинение
    print(return_dict.values())
    for proc in range(procs):
        for v in t_values_list[proc].keys():
            #t_values[v, :] = t_values_list[proc][v]
            #T_values[v] = T_values_list[proc][v]
            t_values[v, :] = return_dict['t'][v]
            T_values[v] = return_dict['T'][v]
    '''

    z_list = [i for i in range(nZ)]
    X_matrixes = []
    print('Вычисляем матрицы:')
    t1 = time.time()
    for z in z_list:
        #print('z =', z)
        X_list = get_X(patient_logs_path, (0, 0, z), nX, nY, nZ, nR, nT, nS);
        X_matrixes.append(X_list.copy())

    t2 = time.time()
    print('Затрачено времени:', t2-t1)
    
    print('Загружаем данные:')
    t3 = time.time()
    patient_data_dict = load_patient_data(patient_data_path)
    t4 = time.time()
    print('Загрузили за', t4-t3)

    print('Начали мультипроцессинг')
    t1 = time.time()
    if __name__ == '__main__':
        p = Pool(processes=procs)
        #computing_func = lambda voxel_list : computing(patient_data_path, \
        #                     patient_logs_path, contrast, K_list, X2, voxel_list, \
        #                     x_list, y_list, z_list, nX, nY, nZ, nR)
        args = [(X_matrixes, \
                             patient_data_dict, contrast, K_list, X2, calcs[proc], \
                             x_list, y_list, z_list, nX, nY, nZ, nR) for proc in range(procs)]
        data = p.starmap(computing, [args_set for args_set in args])
        p.close()
    for proc in range(procs):
        for v in data[proc][0].keys():
            t_values[v, :] = data[proc][0][v]
            T_values[v] = data[proc][1][v]
    t2 = time.time()
    print('Закончили за', t2-t1)
    return t_values, T_values

        

def reproducibility_analysis(patient_data_path, patient_logs_path, contrast, K_list, X2, \
                             x_list=None, y_list=None, z_list=None, nX=None, nY=None, nZ=None, nR=None, nT=None, nS=None):
    nV = nX * nY * nZ
    t0 = time.time()
    if __name__ == "__main__":
        start = time.time()
        # узнаем количество ядер у процессора
        n_proc = multiprocessing.cpu_count()
        # вычисляем сколько циклов вычислений будет приходится
        # на 1 ядро, что бы в сумме получилось 80 или чуть больше
        voxels = []
        for x in range(nX):
            for y in range(nY):
                for z in range(nZ):
                    voxels.append(coord_voxel(x, y, z, nX, nY, nZ))
        #voxels = [i for i in range(nV)]
        calcs = [[] for _ in range(n_proc)]
        for i in range(n_proc):
            calcs[i] = [v for v in voxels if v % n_proc == i]
        print('Начали мультипроцессинг')
        t_values, T_values = multiprocessed(patient_data_path, patient_logs_path, contrast, K_list, X2, \
                                 x_list, y_list, z_list, nX, nY, nZ, nR, \
                            n_proc, calcs)
        end = time.time()
        print(f"Всего {n_proc} ядер в процессоре")
        print(f"На каждом ядре произведено {max([len(calc) for calc in calcs])} циклов вычислений")
        print(f"Итого {n_proc*max([len(calc) for calc in calcs])} циклов за: ", end - start) 
    print('Размер t_values:', t_values.shape)
    print(t_values)
    n_t_dict = {}
    for i in range(t_values.shape[0]):
        t = tuple(t_values[i])
        if t not in n_t_dict.keys():
            n_t_dict[t] = 0
        n_t_dict[t] += 1
    print('Число уникальных наборов t_values:', len(n_t_dict.keys()))
    print(n_t_dict.keys())
    print(n_t_dict)
    t1 = time.time()
    print('Мультипроцессинг за', t1-t0)
    final_thresholds = get_best_thresholds(K_list, t_values, nX, nY, nZ, nR, nT, nS)
    t2 = time.time()
    print('Лучший порог за', t2-t1)
    
    voxel_activity_dict = get_reproducibility(K_list, final_thresholds, t_values)
    voxel_status_dict = get_map_array(K_list, voxel_activity_dict, nX, nY, nZ)
    final_voxel_status_dict = {}
    t3 = time.time()
    print('Воспроизводимость за', t3-t2)
    print('Рисуем для:')
    print(x_list)
    print(y_list)
    print(z_list)
    for K in K_list:
        print(f'Построение графиков для K={K} порогов')
        final_voxel_status_dict[K] = plot_map(voxel_status_dict[K], x_list, y_list, z_list)
    return final_voxel_status_dict


def chunk_voxels(voxels: List[int], n_chunks: int, max_iter=100, time_to_create=80) -> List[List[int]]:
    summ = 0
    time_to_create = int(time_to_create * max_iter / 100)
    i = 1
    while summ < len(voxels):
        summ += i * time_to_create
        i += 1
    n_chunks = min(i - 1, n_chunks)
    sum_time_to_create = sum([i * time_to_create for i in range(n_chunks)])
    chunk_base_size = (len(voxels)+sum_time_to_create) // n_chunks
    chunks_sizes = [chunk_base_size - i * time_to_create for i in range(n_chunks)]
    chunks_sizes[0] += len(voxels) - sum(chunks_sizes)
    print(chunks_sizes)
    voxels_mixed = voxels.copy()
    random.shuffle(voxels_mixed)
    chunks = [[] for _ in range(n_chunks)]
    for i in range(n_chunks):
        chunks[i] = voxels_mixed[:chunks_sizes[i]]
        voxels_mixed = voxels_mixed[chunks_sizes[i]:]
    return chunks



def run_computing_on_chunk(args) -> Tuple[Dict, Dict]:
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=IntegrationWarning)
    warnings.filterwarnings("ignore", message='.*cgi.*', category=IntegrationWarning)
    voxels_chunk, X_matrixes, patient_data_dict, contrasts, K_list, X2, x_list, y_list, z_list, nX, nY, nZ, nR, nT, nS, max_iter, runs = args
    return computing(X_matrixes, patient_data_dict, contrasts, K_list, X2, voxels_chunk, x_list, y_list, z_list, nX, nY, nZ, nR, nT, nS, max_iter=max_iter, runs=runs)




def is_serializable(obj):
    try:
        pickle.dumps(obj)
        return True
    except (pickle.PickleError, TypeError, AttributeError):
        return False



def total_size(obj):
    return sys.getsizeof(pickle.dumps(obj))

class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super(NpEncoder, self).default(obj)




#nX = 5
#nY = 5
#nZ = 3
#voxels = []
#for x in range(nX):
#    for y in range(nY):
#        for z in range(nZ):
#            print(x, y, z)
#            voxels.append(coord_voxel(x, y, z, nX, nY, nZ))
#            print(voxels[-1])
#            print(voxel_coord(voxels[-1], nX, nY, nZ))
#            print()



























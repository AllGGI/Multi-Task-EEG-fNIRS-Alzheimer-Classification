import scipy
from scipy import io
import os
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import pickle
import pandas as pd
from tqdm import tqdm
import copy

from scipy.signal import butter, filtfilt
from scipy.stats import kurtosis, skew



def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def statistical_features(ch_data):

    bandpassed = butter_bandpass_filter(ch_data, lowcut=0.01, highcut=0.2, fs=8, order=5)

    mean = np.mean(bandpassed)
    std = np.std(bandpassed)
    sums = np.sum(bandpassed)
    
    return [mean, std, sums]


def simple_processing_fnirs(levels_list):
    
    fnirs_sampling_rate = 8
    # fnirs_band = [[0.01, 0.1]]
    fnirs_ch = 6

    patient_out = []
    for label, level_data in enumerate(levels_list): # ad 26 -> norm 64 -> mci 46
        for patient in level_data: # 26
            act_out = [[] for i in range(6)]
            for act_num, act_data in enumerate(patient): # 6 = RO, C1, C2, N1, N2, V
                if act_num == 0: # RO
                    hb_data = act_data['Hb'][:fnirs_sampling_rate*60, :] # RO - 60 sec (480, 6)
                    hbo_data = act_data['HbO'][:fnirs_sampling_rate*60, :] # RO - 60 sec (480, 6)
                    thb_data = act_data['THb'][:fnirs_sampling_rate*60, :] # RO - 60 sec (480, 6)

                    hb_resting = [statistical_features(hb_data[:,ch]) for ch in range(6)]
                    hb_resting = np.stack(hb_resting, axis=0)
                    hbo_resting = [statistical_features(hbo_data[:,ch]) for ch in range(6)]
                    hbo_resting = np.stack(hbo_resting, axis=0)
                    thb_resting = [statistical_features(thb_data[:,ch]) for ch in range(6)]
                    thb_resting = np.stack(thb_resting, axis=0)

                    all_hb_features = np.stack([hb_resting, hbo_resting, thb_resting], axis=0) # (3hb, 6, 3)

                    act_out[act_num] = all_hb_features



                else: # C, N (2299, 6)
                    hb_data_dict = act_data['Hb'] # 'before','ing','after'
                    hbo_data_dict = act_data['HbO']
                    thb_data_dict = act_data['THb']

                    # (240, 6) (1819, 6) (240, 6) -> (6,3), (6,3), (6,3)
                    # print(hb_data_dict['before'].shape, hb_data_dict['ing'].shape, hb_data_dict['after'].shape)
                    hb_before = [statistical_features(hb_data_dict['before'][:,ch]) for ch in range(6)]
                    hb_before = np.stack(hb_before, axis=0)
                    hb_ing = [statistical_features(hb_data_dict['ing'][:,ch]) for ch in range(6)]
                    hb_ing = np.stack(hb_ing, axis=0)
                    hb_after = [statistical_features(hb_data_dict['after'][:,ch]) for ch in range(6)]
                    hb_after = np.stack(hb_after, axis=0)
                    hb_features = np.stack([hb_before, hb_ing, hb_after], axis=0) # (3=bef,ing,aft, 6ch, 3=mean,std,sum)

                    hbo_before = [statistical_features(hbo_data_dict['before'][:,ch]) for ch in range(6)]
                    hbo_before = np.stack(hbo_before, axis=0)
                    hbo_ing = [statistical_features(hbo_data_dict['ing'][:,ch]) for ch in range(6)]
                    hbo_ing = np.stack(hbo_ing, axis=0)
                    hbo_after = [statistical_features(hbo_data_dict['after'][:,ch]) for ch in range(6)]
                    hbo_after = np.stack(hbo_after, axis=0)
                    hbo_features = np.stack([hbo_before, hbo_ing, hbo_after], axis=0) # (3=bef,ing,aft, 6ch, 3=mean,std,sum)

                    thb_before = [statistical_features(thb_data_dict['before'][:,ch]) for ch in range(6)]
                    thb_before = np.stack(thb_before, axis=0)
                    thb_ing = [statistical_features(thb_data_dict['ing'][:,ch]) for ch in range(6)]
                    thb_ing = np.stack(thb_ing, axis=0)
                    thb_after = [statistical_features(thb_data_dict['after'][:,ch]) for ch in range(6)]
                    thb_after = np.stack(thb_after, axis=0)
                    thb_features = np.stack([thb_before, thb_ing, thb_after], axis=0) # (3=bef,ing,aft, 6ch, 3=mean,std,sum)

                    all_hb_features = np.stack([hb_features, hbo_features, thb_features], axis=0) # (3hb, 3, 6, 3)

                    act_out[act_num] = all_hb_features

            patient_out.append([label, act_out])


    # patient_out[patient_idx][0] = label | patient_out[patient_idx][1] = act_out
    # act_out[0] = data[3 hb][6 ch][3 mean/std/med]
    # act_out[1~5] = data[3 hb][3 bef/ing/aft][6 ch][3 mean/std/med]
    flattened_data = []
    for i in range(len(patient_out)):
        label = patient_out[i][0]
        one_patient_data = patient_out[i][1]
        all_act_data = []
        for act_num in range(6):
            fnirs_data = one_patient_data[act_num]

            if act_num == 0: # 'Resting' = data[3 hb][6 ch][3 mean/std/med] = 3*6*3 = 54
                all_act_data += [fnirs_data[hb_type][ch][feat] for hb_type in range(3) for ch in range(6) for feat in range(3)]
            else: # Others = data[3 hb][3 bef/ing/aft][6 ch][3 mean/std/med] = 3*3*6*3 = 162
                all_act_data += [fnirs_data[hb_type][time_type][ch][feat] for hb_type in range(3) for time_type in range(3) for ch in range(6) for feat in range(3)]

        flattened_data.append([label] + all_act_data)

    return flattened_data # flattened 'patient_all_data_normalized'


def timedomain2csv():
    
    sliced_pth = './pickles/original_fnirs_before_ing_after.pkl'
    new_AD_sliced_pth = './pickles/new-AD_fnirs_before_ing_after.pkl'


    # if not os.path.exists('./inputs/sliced/'):
    #     os.makedirs('./inputs/sliced/')

    with open(sliced_pth, 'rb') as f:
        loaded_data = pickle.load(f) # ad, cn, ns, pre order
        print('>> ' + sliced_pth + ' loaded.')

    levels = ['AD', 'NORMAL', 'MCI']

    ad_fnirs = loaded_data[0]['fNIRs']
    norm_fnirs = loaded_data[1]['fNIRs']
    mci_fnirs = loaded_data[3]['fNIRs']


    with open(new_AD_sliced_pth, 'rb') as f:
        new_loaded_data = pickle.load(f) # ad, cn, ns, pre order
        print('>> ' + new_AD_sliced_pth + ' loaded.')    

    ad_fnirs += new_loaded_data[0]['fNIRs']

    print(len(ad_fnirs), len(norm_fnirs), len(mci_fnirs))
    all_levels_fnirs = [ad_fnirs, norm_fnirs, mci_fnirs]

    flatten_fnirs = simple_processing_fnirs(all_levels_fnirs)

    ####################################################################################################
    # 5) making column
    # act_out[0] = data[3 hb][6 ch][3 mean/std/med]
    # act_out[1~5] = data[3 hb][3 bef/ing/aft][6 ch][3 mean/std/med]
    column = ['label']
    hb_types = ['Hb', 'HbO','THb']
    time_types = ['before', 'ing', 'after']
    feat_types = ['mean', 'std', 'sum']

    # 1(label) + 54 + 162*5 = 865
    for act_num in range(6):
        if act_num == 0: # 54
            for hb_type in hb_types:
                for ch in range(6):
                    for feat_type in feat_types:
                        column += ['fnirs_act-{}_prop-{}_ch-{}_feat-{}'.format(act_num, hb_type, ch, feat_type)]
        else: # 162
            for hb_type in hb_types:
                for time_type in time_types: 
                    for ch in range(6):
                        for feat_type in feat_types:
                            column += ['fnirs_act-{}_prop-{}_time-{}_ch-{}_feat-{}'.format(act_num, hb_type, time_type, ch, feat_type)]

    ####################################################################################################
    # 6) save csv
    # make save folder
    if not os.path.exists('./csv_folder/source/'):
        os.makedirs('./csv_folder/source/')
    df = pd.DataFrame(flatten_fnirs, columns=column)
    df.to_csv('./csv_folder/source/Source3-fnirs_time_domain_features.csv', sep=',')
    print('>> csv saved.')

    print(df)



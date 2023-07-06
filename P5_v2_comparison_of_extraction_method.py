import os
import numpy as np
import pandas as pd
import pickle
from time import time
import spkit as sp

# My files
from P4_v1_make_index import make_index
from P4_v1_get_intersection_of_optimal_features import get_intersection_of_folds

def multiple_acts(total_X, total_y, act_list=[0,1,2,3,4,5], norm=True): # 4752 / 6 -> 792

    # # Select desired act cols
    # only_this_act_cols = []
    # for col in total_X.columns:
    #     act_part = int(col.split('_')[1][-1]) # 0-5
        
    #     if act_part in act_list: only_this_act_cols.append(col)

    only_this_act_cols = list(total_X.columns)
    

    # Split EEG / fNIRS col
    fnirs_start_idx = -1
    for idx, col in enumerate(only_this_act_cols):
        if 'fnirs' in col:
            fnirs_start_idx = idx
            break
    
    eeg_features = only_this_act_cols[:fnirs_start_idx]
    fnirs_features = only_this_act_cols[fnirs_start_idx:]

    print(fnirs_start_idx)
    print(len(eeg_features))
    print(len(fnirs_features))

    ret = {}
    X_eeg = total_X[eeg_features]
    X_fnirs = total_X[fnirs_features]
    y_eeg = total_y
    y_fnirs = total_y

    if norm:
        # Min-Max Normalization
        X_eeg = X_eeg.apply(lambda x: (x - x.min()) / (x.max() - x.min()), axis=0)
        X_fnirs = X_fnirs.apply(lambda x: (x - x.min()) / (x.max() - x.min()), axis=0)


    ret['X_eeg'] = X_eeg
    ret['X_fnirs'] = X_fnirs
    ret['y_eeg'] = y_eeg
    ret['y_fnirs'] = y_fnirs

    print('>> Original features num (EEG, FNIRS):', len(eeg_features), len(fnirs_features))

    return ret


def load_pickle():

    slice_sec_num = 10
    sliced_pth = './pickles/eeg-' + str(slice_sec_num) + 'sec_fnirs-full_baselinecorrected_fnirs_sliced.pkl'
    new_AD_sliced_pth = './pickles/new-AD_eeg-' + str(slice_sec_num) + 'sec_fnirs-full_baselinecorrected_fnirs_sliced.pkl'

    with open(sliced_pth, 'rb') as f:
        loaded_data = pickle.load(f) # ad, cn, ns, pre order
        print('>> ' + sliced_pth + ' loaded.')

    ad_eeg = loaded_data[0]['EEG']
    norm_eeg = loaded_data[1]['EEG']
    mci_eeg = loaded_data[3]['EEG']

    ad_fnirs = loaded_data[0]['fNIRs']
    norm_fnirs = loaded_data[1]['fNIRs']
    mci_fnirs = loaded_data[3]['fNIRs']


    with open(new_AD_sliced_pth, 'rb') as f:
        new_loaded_data = pickle.load(f) # ad, cn, ns, pre order
        print('>> ' + new_AD_sliced_pth + ' loaded.')    
    ad_eeg += new_loaded_data[0]['EEG']
    ad_fnirs += new_loaded_data[0]['fNIRs']

    print(len(ad_eeg), len(norm_eeg), len(mci_eeg))
    print(len(ad_fnirs), len(norm_fnirs), len(mci_fnirs))

    eeg_list = [ad_eeg, norm_eeg, mci_eeg]
    fnirs_list = [ad_fnirs, norm_fnirs, mci_fnirs]
    
    return eeg_list, fnirs_list


def comparison1_eeg(eeg_list, levels, input_pth):
    '''
    3. Feature extraction
        0) Block average? => Average all trials of each tasks (= diff sequences)
        1) EEG (total 10 sec)
            - 2 second time window * 5 segment (no overlap)
            - for each time window, for each band:
                Feature = Relative Band Power = Each band power / Sum(all band Power)
                (Band power = PSD w/ FFT)
            - 33 channel * 6 band * 5 time window * 4 block = 3960 features
    '''
    
    eeg_sampling_rate = 500
    eeg_channels = 32
    eeg_bands = [[0.5,4],[4,7],[8,10],[10,13],[14,25],[26,40]] # EEG bands of previous study

    for level_idx, level_data in enumerate(eeg_list): # ad 26 -> norm 64 -> mci 46
        patients = []
        for patient in level_data: # 26
            act_out = [[] for i in range(6)]
            for act_num, act_data in enumerate(patient): # 6 = RO, C1, C2, N1, N2, V
                if act_num == 0:
                    rest_segments = []
                    for seg_idx in range(6):
                        rest_seg = act_data[5000*seg_idx:5000*(seg_idx+1), :]
                        rest_segments.append(rest_seg)
                    act_data = rest_segments

                # if act_num == 3 or act_num == 4: # C, N (1500=[-1,2 sec], 32)
                n_seg = 1000
                n_overlap = 0
                
                act_data = np.stack(act_data, axis=0) # (32, 5000, 32)
                block_avg = np.mean(act_data, axis=0) # avg of each 10 sec (5000, 32)

                this_act_out = []
                for i in range(5):
                    time_window_2sec = block_avg[1000*i:1000*(i+1), :]
                    Px, _, _ = sp.eeg.eeg_processing.RhythmicDecomposition(E=time_window_2sec, fBands=eeg_bands, fs=eeg_sampling_rate, nperseg=n_seg, noverlap=n_overlap, Sum=True, Mean=False, SD=False)
                    
                    sum_Px = np.sum(Px, axis=0).reshape(1,-1)
                    RBP = Px/sum_Px # Relative Band Power (6,32)
                    
                    this_act_out.append(RBP.T) # (32, 6)

                this_act_out = np.stack(this_act_out, axis=-1) # (32, 6, 5)
                act_out[act_num] = this_act_out # (32, 6, 5)

            act_out = np.stack(act_out, axis=-1) # (32ch, 6 band, 5 window, 6 block -> 6 act)
            patients.append(act_out)
        
        patients = np.stack(patients, axis=0)
        print(levels[level_idx], patients.shape) # AD (35, 32, 6, 5, 2)

        np.save(input_pth + levels[level_idx] + '_eeg_RBP.npy', patients)
        print('>> npy saved.')

def comparison1_fnirs(fnirs_list, levels, input_pth):
    '''
    3. Feature extraction
        0) Block average? => Average all trials of each tasks (= diff sequences)
        2) fNIRS (3s ~ 12s = total 9 sec)
            - 3 second time window * 3 segment (no overlap)
            - Feature = Avg changes of HbO, HbR concentrations
            (By taking avg of data every 3 s with no overlapping for block averaged signals.)
            - 46 channel * 2 Hb * 3 time window * 4 block = 1104 features
    
    '''
    fnirs_window_num = 3
    fnirs_sampling_rate = 8
    fnirs_channels = 6
    for level_idx, level_data in enumerate(fnirs_list): # ad 26 -> norm 64 -> mci 46
        patients = []
        for patient_idx, patient in enumerate(level_data): # 26
            act_out = [[] for i in range(6)]
            for act_num, act_data in enumerate(patient): # 6 = RO, C1, C2, N1, N2, V

                if act_num == 0:
                    hb_data_list, hbo_data_list = [], []
                    for seg_idx in range(6):
                        hb_seg = act_data['Hb'][80*seg_idx:(80*seg_idx)+72, :]
                        hbo_seg = act_data['HbO'][80*seg_idx:(80*seg_idx)+72, :]
                        hb_data_list.append(hb_seg)
                        hbo_data_list.append(hbo_seg)

                else:
                    hb_data_list = act_data['Hb']
                    hbo_data_list = act_data['HbO']

                # hb_data_list = [hb[:,:] for hb in hb_data_list] # 1sec~10sec = 9s
                # hbo_data_list = [hbo[:,:] for hbo in hbo_data_list] # 1sec~10sec = 9s
                
                hb_data_list = np.stack(hb_data_list, axis=0)
                hb_block_avg = np.mean(hb_data_list, axis=0)
                hb_time_windows = [np.mean(hb_block_avg[24*i:24*(i+1), :], axis=0) for i in range(fnirs_window_num)] # 3 sec * 3 window
                hb_time_windows = np.stack(hb_time_windows, axis=-1) # (6 ch, 2 windows)

                hbo_data_list = np.stack(hbo_data_list, axis=0)
                hbo_block_avg = np.mean(hbo_data_list, axis=0)
                hbo_time_windows = [np.mean(hbo_block_avg[24*i:24*(i+1), :], axis=0) for i in range(fnirs_window_num)] # 3 sec * 3 window
                hbo_time_windows = np.stack(hbo_time_windows, axis=-1) # (6 ch, 2 windows)

                this_act_out = np.stack([hb_time_windows, hbo_time_windows], axis=1)
                act_out[act_num] = this_act_out

            act_out = np.stack(act_out, axis=-1) # (6ch, 2 hb, 3 window, 6 block -> 6 act)
            patients.append(act_out)
        
        patients = np.stack(patients, axis=0)
        print(levels[level_idx], patients.shape) # AD (35, 6, 2, 3, 2)

        np.save(input_pth + levels[level_idx] + '_fnirs_avg.npy', patients)
        print('>> npy saved.')

def comparison1_data2csv(levels, input_pth, output_path):
    fnirs_window_num = 3

    eeg_npy, fnirs_npy = {}, {}
    for level in levels:
        eeg_npy[level] = np.load(input_pth + level + '_eeg_RBP.npy')
        fnirs_npy[level] = np.load(input_pth + level + '_fnirs_avg.npy')

    # EEG: (35 patients, 32 ch, 6 band, 5 window, 6 block)
    eeg_cols = []
    eeg_bands = ['Delta', 'Theta', 'LowAlpha', 'HighAlpha', 'Beta', 'Gamma']
    for ch in range(32):
        for band in range(len(eeg_bands)):
            for window in range(5):
                for block in range(6): # block == act num
                    col_name = 'eeg_ch-' + str(ch) + '_band-' + eeg_bands[band] + '_window-' + str(window) + '_block-' + str(block)
                    eeg_cols.append(col_name)

    print(len(eeg_cols)) # 1920

    # fNIRS: (35 patients, 6 ch, 2 hb, 2 window, 6 block)
    fnirs_cols = []
    fnirs_hb = ['Hb', 'HbO']
    for ch in range(6):
        for h in range(len(fnirs_hb)):
            for window in range(fnirs_window_num):
                for block in range(6): # block == act num
                    col_name = 'fnirs_ch-' + str(ch) + '_' + fnirs_hb[h] + '_window-' + str(window) + '_block-' + str(block)
                    fnirs_cols.append(col_name)

    print(len(fnirs_cols)) # 72

    ################################################################################
    # EEG
    EEG_all_levels = []
    for level in levels:
        level_flatten_data = []
        for patient in range(eeg_npy[level].shape[0]):
            flatten_data = []
            this_patient_data = eeg_npy[level][patient]
            for ch in range(32):
                for band in range(len(eeg_bands)):
                    for window in range(5):
                        for block in range(6): # block == act num
                            flatten_data.append(this_patient_data[ch][band][window][block])
            
            level_flatten_data.append(flatten_data)

        level_flatten_data = np.asarray(level_flatten_data)
        print(level_flatten_data.shape)
        EEG_all_levels.append(level_flatten_data)

    ################################################################################
    # FNIRS
    FNIRS_all_levels = []
    for level in levels:
        level_flatten_data = []
        for patient in range(fnirs_npy[level].shape[0]):
            flatten_data = []
            this_patient_data = fnirs_npy[level][patient]
            for ch in range(6):
                for h in range(len(fnirs_hb)):
                    for window in range(fnirs_window_num):
                        for block in range(6): # block == act num
                            flatten_data.append(this_patient_data[ch][h][window][block])
            
            level_flatten_data.append(flatten_data)

        level_flatten_data = np.asarray(level_flatten_data)
        print(level_flatten_data.shape) # (26, 1296)
        FNIRS_all_levels.append(level_flatten_data)

    all_levels = [np.concatenate([EEG_all_levels[level], FNIRS_all_levels[level]], axis=1) for level in range(len(levels))]
    print(all_levels[0].shape, all_levels[1].shape, all_levels[2].shape) # (?, 1920+72=1992)
    

    all_input = np.concatenate(all_levels, axis=0) # (35+64+46=144, 1992)
    labels = [0] * len(all_levels[0]) + [1] * len(all_levels[1]) + [2] * len(all_levels[2])
    all_cols = eeg_cols + fnirs_cols

    assert all_input.shape[0] == len(labels) and all_input.shape[1] == len(all_cols)

    df = pd.DataFrame(all_input, columns=all_cols)
    df.insert(0, 'label', labels)
    print(df) # (145 x 1993)
    print('*'*100)

    # df = df[np.isfinite(df).all(1)]
    df.to_csv(output_path + 'comparison1_eegfnirs_allact.csv', sep=',')
    print(output_path + 'comparison1_eegfnirs_allact.csv saved.')


def my_eeg_features():
    open_csv = './csv_folder/Experiment2/opt-5_5secEEGPSD.csv'

    df = pd.read_csv(open_csv)
    df = df.drop('Unnamed: 0', axis=1)
    # df = df[np.isfinite(df).all(1)]
    print(df)

    my_EEG_X = df.drop('label', axis=1)
    my_EEG_y = df['label']

    return my_EEG_X, my_EEG_y


def my_fnirs_features():
    open_csv = './csv_folder/Experiment2/opt-4_FullFnirsPSD_FullFnirsTimeDomain.csv'
    df = pd.read_csv(open_csv)
    df = df.drop('Unnamed: 0', axis=1)

    df.drop([72], axis=0, inplace=True)
    # df = df[np.isfinite(df).all(1)]

    print(df)

    my_fnirs_X = df.drop('label', axis=1)
    my_fnirs_y = df['label']

    return my_fnirs_X, my_fnirs_y

def comparison1_loadcsv(load_pth, act_num):
    open_csv = load_pth + 'comparison1_eegfnirs_allact.csv'
    df = pd.read_csv(open_csv)
    df = df.drop('Unnamed: 0', axis=1)
    
    df.drop([72], axis=0, inplace=True)
    # df = df[np.isfinite(df).all(1)]
    
    print(df)

    comp_X = df.drop('label', axis=1)
    comp_y = df['label']

    data_dict = multiple_acts(comp_X, comp_y, act_list=act_num, norm=True)

    return data_dict['X_eeg'], data_dict['X_fnirs'] # 5760, 216



def extraction_comparison2csv(act_num):
    
    levels = ['AD', 'NORMAL', 'MCI']
    input_path = './inputs/Experiment3/'
    output_path = './csv_folder/Experiment3/'

    if not os.path.exists(input_path):
        os.makedirs(input_path)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # 1) Make CSV of My data + Previous method
    E_list, f_list = load_pickle()
    comparison1_eeg(E_list, levels, input_path)
    comparison1_fnirs(f_list, levels, input_path)
    comparison1_data2csv(levels, input_path, output_path)

    # 2) Load My EEG/fNIRS + Previous EEG/fNIRS
    comp_X_eeg, comp_X_fnirs = comparison1_loadcsv(output_path, act_num)
    my_X_eeg, my_y_eeg = my_eeg_features()
    my_X_fnirs, my_y_fnirs = my_fnirs_features()

    assert my_y_eeg.to_list() == my_y_fnirs.to_list()

    # 3) My EEG extraction + Previous fNIRS extraction
    MyEEG_PrevfNIRS = pd.concat([my_X_eeg, comp_X_fnirs], axis=1)
    MyEEG_PrevfNIRS.insert(0, 'label', my_y_eeg)
    MyEEG_PrevfNIRS = MyEEG_PrevfNIRS[np.isfinite(MyEEG_PrevfNIRS).all(1)]

    # 4) Previous EEG extraction + My fNIRS extraction
    PrevEEG_MyfNIRS = pd.concat([comp_X_eeg, my_X_fnirs], axis=1)
    PrevEEG_MyfNIRS.insert(0, 'label', my_y_eeg)
    PrevEEG_MyfNIRS = PrevEEG_MyfNIRS[np.isfinite(PrevEEG_MyfNIRS).all(1)]

    # 5) Previous EEG extraction + Previous fNIRS extraction
    PrevEEG_PrevfNIRS = pd.concat([comp_X_eeg, comp_X_fnirs], axis=1)
    PrevEEG_PrevfNIRS.insert(0, 'label', my_y_eeg)
    PrevEEG_PrevfNIRS = PrevEEG_PrevfNIRS[np.isfinite(PrevEEG_PrevfNIRS).all(1)]

    # 6) Make csvs
    MyEEG_PrevfNIRS.to_csv(output_path + 'MyEEG_PrevfNIRS.csv', sep=',')
    PrevEEG_MyfNIRS.to_csv(output_path + 'PrevEEG_MyfNIRS.csv', sep=',')
    PrevEEG_PrevfNIRS.to_csv(output_path + 'PrevEEG_PrevfNIRS.csv', sep=',')


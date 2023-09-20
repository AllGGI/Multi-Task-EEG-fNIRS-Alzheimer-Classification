###################################################################
# fnirs PSD + 5sec EEG PSD + fnirs time domain csv
###################################################################

import pandas as pd
import numpy as np
import os


def multiple_acts(total_X, total_y, act_list=[], norm=True): # 4752 / 6 -> 792

    # print(len(total_X.columns)) # 4752
    
    only_this_act_cols = []
    for col in total_X.columns:
        act_part = int(col.split('_')[1][-1]) # 0-5
        
        if act_part in act_list: only_this_act_cols.append(col)

    # print(len(only_this_act_cols)) # 792

    fnirs_start_idx = -1
    for idx, col in enumerate(only_this_act_cols):
        if 'fnirs' in col:
            fnirs_start_idx = idx
            break
    
    eeg_features = only_this_act_cols[:fnirs_start_idx]
    fnirs_features = only_this_act_cols[fnirs_start_idx:]

    print('>> # EEG features:', len(eeg_features))
    print('>> # fNIRS features:', len(fnirs_features))

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

    return ret


def multimodal2csv():

    act_num = [0,1,2,3,4,5] # Resting=0, C1=1, C2=2, N1=3, N2=4, V=5
    time_sec = 5

    # source files to make multi-modal csv
    eeg_5sec_psd_csv = './csv_folder/source/Source1-' + str(time_sec) + 'sec_eeg_fnirs_power_sliced.csv'
    full_fnirs_psd_csv = './csv_folder/source/Source2-full_eeg_fnirs_psd.csv'
    full_fnirs_timedomain_csv = './csv_folder/source/Source3-fnirs_time_domain_features.csv'
    

    # 1) 5sec EEG PSD
    eeg_5sec_psd_df = pd.read_csv(eeg_5sec_psd_csv)
    eeg_5sec_psd_df = eeg_5sec_psd_df.drop('Unnamed: 0', axis=1)
    # df = df[np.isfinite(df).all(1)]
    X_eeg_5sec_psd_df = eeg_5sec_psd_df.drop('label', axis=1)
    y_eeg_5sec_psd_df = eeg_5sec_psd_df['label']

    temp_dict = multiple_acts(X_eeg_5sec_psd_df, y_eeg_5sec_psd_df, act_list=act_num, norm=True)
    needed_X_eeg_psd = temp_dict['X_eeg']

    # 2) Full fNIRS PSD
    full_fnirs_psd_df = pd.read_csv(full_fnirs_psd_csv)
    full_fnirs_psd_df = full_fnirs_psd_df.drop('Unnamed: 0', axis=1)
    X_full_fnirs_psd_df = full_fnirs_psd_df.drop('label', axis=1)
    y_full_fnirs_psd_df = full_fnirs_psd_df['label']

    temp_dict = multiple_acts(X_full_fnirs_psd_df, y_full_fnirs_psd_df, act_list=act_num, norm=True)
    needed_X_fnirs_psd = temp_dict['X_fnirs']

    # 3) fNIRS Time Domain
    full_fnirs_timedomain_df = pd.read_csv(full_fnirs_timedomain_csv)
    full_fnirs_timedomain_df = full_fnirs_timedomain_df.drop('Unnamed: 0', axis=1)
    X_full_fnirs_timedomain_df = full_fnirs_timedomain_df.drop('label', axis=1)
    y_full_fnirs_timedomain_df = full_fnirs_timedomain_df['label']

    temp_dict = multiple_acts(X_full_fnirs_timedomain_df, y_full_fnirs_timedomain_df, act_list=act_num, norm=True)
    needed_X_fnirs_timedomain = temp_dict['X_fnirs']

    assert len(eeg_5sec_psd_df) == len(full_fnirs_psd_df) == len(full_fnirs_timedomain_df)
    assert y_eeg_5sec_psd_df.tolist() == y_full_fnirs_psd_df.tolist() == y_full_fnirs_timedomain_df.tolist()



    # Options
    # 1: EEG PSD + fnirs PSD + fnirs time
    # 2: EEG PSD + fnirs PSD
    # 3: EEG PSD + fnirs time
    # 4: fnirs PSD + fnirs time
    # 5: EEG PSD
    for option in range(1,4):
        print('\n>> Current option:', option)
        if option == 1: # EEG + fNIRS
            complex_df = pd.concat([needed_X_eeg_psd, needed_X_fnirs_psd, needed_X_fnirs_timedomain], axis=1)
            save_csv_name = str(time_sec) + 'secEEGPSD_FullFnirsPSD_FullFnirsTimeDomain_NEW'
        elif option == 2:
            complex_df = pd.concat([needed_X_eeg_psd], axis=1)
            save_csv_name = 'opt-' + str(option) + '_' + str(time_sec) + 'secEEGPSD'
        elif option == 3: # fNIRS
            complex_df = pd.concat([needed_X_fnirs_psd, needed_X_fnirs_timedomain], axis=1)
            save_csv_name = 'opt-' + str(option) + '_' + 'FullFnirsPSD_FullFnirsTimeDomain'
        # elif option == 4:
        #     complex_df = pd.concat([needed_X_eeg_psd, needed_X_fnirs_psd], axis=1)
        #     save_csv_name = 'opt-' + str(option) + '_' + str(time_sec) + 'secEEGPSD_FullFnirsPSD'
        # elif option == 5:
        #     complex_df = pd.concat([needed_X_eeg_psd, needed_X_fnirs_timedomain], axis=1)
        #     save_csv_name = 'opt-' + str(option) + '_' + str(time_sec) + 'secEEGPSD_FullFnirsTimeDomain'
        
        complex_df.insert(0, 'label', y_eeg_5sec_psd_df)
        complex_df = complex_df.dropna(axis=0)

        if not os.path.exists('./csv_folder/Experiment2/'):
            os.makedirs('./csv_folder/Experiment2/')
        
        complex_df.to_csv('./csv_folder/Experiment2/' + save_csv_name + '.csv', sep=',')
        print(save_csv_name + '.csv saved.')



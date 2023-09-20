import numpy as np
import pandas as pd
import os

global_slice_sec = 10

levels = ['AD', 'NORMAL', 'MCI']
features = ['Px', 'Pm', 'Pd']
fnirs_hb = ['Hb', 'HbO', 'THb']

eeg_bands = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma1', 'Gamma2']
fnirs_bands = ['vlfo', 'lfo'] # [[0.01, 0.04], [0.04, 0.15]]


def make_cols():

    eeg_cols = []
    eeg_channels = 32
    for act in range(6):
        for p in range(len(features)):
            for band in range(len(eeg_bands)):
                for ch in range(32):
                    col_name = 'eeg_act-' + str(act) + '_feat-' + features[p] + '_band-' + eeg_bands[band] + '_ch-' + str(ch)
                    eeg_cols.append(col_name)

    print(len(eeg_cols)) # 3456

    fnirs_cols = []
    fnirs_channels = 6
    for act in range(6): # (6act, 3hb, 3px, 4band, 6ch)
        for h in range(len(fnirs_hb)):
            for p in range(len(features)):
                for band in range(len(fnirs_bands)):
                    for ch in range(6):
                        col_name = 'fnirs_act-' + str(act) + '_' + fnirs_hb[h] + '_feat-' + features[p] + '_band-' + fnirs_bands[band] + '_ch-' + str(ch)
                        fnirs_cols.append(col_name)

    print(len(fnirs_cols)) # 1296

    return eeg_cols, fnirs_cols



def sliced2csv(eeg_cols, fnirs_cols, load_pth, save_csv_name):

    # EEG
    EEG_all_levels = []
    for level in levels:
        eeg_file = np.load(load_pth + level + '_eeg_power_SMSD.npy')
        # print(eeg_file.shape) # (patients, 6 act, 3 - Px,Pm,Pd, 6 band, 32 ch)
        level_flatten_data = []
        for patient in range(eeg_file.shape[0]):
            flatten_data = []
            this_patient_data = eeg_file[patient] # (6 act, 3 - Px,Pm,Pd, 6 band, 32 ch))
            for act in range(6):
                for p in range(len(features)):
                    for band in range(len(eeg_bands)):
                        for ch in range(32):
                            flatten_data.append(this_patient_data[act][p][band][ch])
            
            level_flatten_data.append(flatten_data)

        level_flatten_data = np.asarray(level_flatten_data)
        print(level_flatten_data.shape)
        EEG_all_levels.append(level_flatten_data)

    ################################################################################
    # FNIRS
    FNIRS_all_levels = []
    for level in levels:
        fnirs_file = np.load(load_pth + level + '_fnirs_power_SMSD.npy')
        print(fnirs_file.shape) # (patients, 6act, 3hb, 3px, 4band, 6ch)
        
        level_flatten_data = []
        for patient in range(fnirs_file.shape[0]):
            flatten_data = []
            this_patient_data = fnirs_file[patient] # (6act, 3hb, 3px, 4band, 6ch)
            for act in range(6): # (6act, 3hb, 3px, 4band, 6ch)
                for h in range(len(fnirs_hb)):
                    for p in range(len(features)):
                        for band in range(len(fnirs_bands)):
                            for ch in range(6):
                                flatten_data.append(this_patient_data[act][h][p][band][ch])
            
            level_flatten_data.append(flatten_data)

        level_flatten_data = np.asarray(level_flatten_data)
        print(level_flatten_data.shape) # (26, 1296)
        FNIRS_all_levels.append(level_flatten_data)

    all_levels = [np.concatenate([EEG_all_levels[level], FNIRS_all_levels[level]], axis=1) for level in range(len(EEG_all_levels))]
    # print(all_levels[0].shape, all_levels[1].shape, all_levels[2].shape) # (?, 4752)

    all_input = np.concatenate(all_levels, axis=0) # (136, 4752)
    labels = [0] * len(all_levels[0]) + [1] * len(all_levels[1]) + [2] * len(all_levels[2])
    all_cols = eeg_cols + fnirs_cols

    print(all_input.shape) # (136, 4752)
    print(len(labels), len(all_cols)) # 136, 4752

    df = pd.DataFrame(all_input, columns=all_cols)
    df.insert(0, 'label', labels)
    print(df)
    print('*'*100)

    zero_cols = []
    for col in all_cols:
        if sum(df[col]) == 0: zero_cols.append(col)
    print(zero_cols)
    df = df.drop(columns=zero_cols)
    print(df) # 4752
    print('*'*100)

    # print(df[df.isna().any(axis=1)]) # 72
    # # print(df.iloc[72])
    # df.drop([72], axis=0, inplace=True)
    # print(df[df.isna().any(axis=1)])

    # # Remove NA
    # clean_df = df.dropna(axis=0)
    # print(clean_df)
    # print('*'*100)
    # # Normalize
    # X = clean_df.drop('label', axis=1)
    # y = clean_df['label']
    # X = X.apply(lambda x: (x - x.min()) / (x.max() - x.min()), axis=0)
    # X.insert(0, 'label', y)

    if not os.path.exists('./csv_folder/source/'):
        os.makedirs('./csv_folder/source/')
    df.to_csv('./csv_folder/source/' + save_csv_name + '.csv', sep=',')
    print(save_csv_name, 'saved.')
    print(df)

    # print(X)


def psd2csv():

    eeg_cols, fnirs_cols = make_cols()
    sliced2csv(eeg_cols, fnirs_cols, load_pth='./inputs/' + str(global_slice_sec) + '-sec/', save_csv_name='Source1-' + str(global_slice_sec) + 'sec_eeg_fnirs_power_sliced')
    sliced2csv(eeg_cols, fnirs_cols, load_pth='./inputs/full/', save_csv_name='Source2-full_eeg_fnirs_psd')


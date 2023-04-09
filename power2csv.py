import numpy as np
import pandas as pd
import os

levels = ['AD', 'NORMAL', 'MCI']
features = ['Px', 'Pm', 'Pd']

eeg_cols = []
eeg_bands = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma1', 'Gamma2']
eeg_channels = 32
for act in range(6):
    for p in range(3):
        for band in range(6):
            for ch in range(32):
                col_name = 'eeg_act-' + str(act) + '_feat-' + features[p] + '_band-' + eeg_bands[band] + '_ch-' + str(ch)
                eeg_cols.append(col_name)

print(len(eeg_cols)) # 3456

fnirs_cols = []
fnirs_bands = ['vlfo', 'lfo', 'respi', 'heartbeat']
fnirs_hb = ['Hb', 'HbO', 'THb']
fnirs_channels = 6
for act in range(6): # (6act, 3hb, 3px, 4band, 6ch)
    for h in range(3):
        for p in range(3):
            for band in range(4):
                for ch in range(6):
                    col_name = 'fnirs_act-' + str(act) + '_' + fnirs_hb[h] + '_feat-' + features[p] + '_band-' + fnirs_bands[band] + '_ch-' + str(ch)
                    fnirs_cols.append(col_name)

print(len(fnirs_cols)) # 1296



def sliced2csv(load_pth, save_csv_name):

    ################################################################################
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
                for p in range(3):
                    for band in range(6):
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
        # print(fnirs_file.shape) # (patients, 6act, 3hb, 3px, 4band, 6ch)
        
        level_flatten_data = []
        for patient in range(fnirs_file.shape[0]):
            flatten_data = []
            this_patient_data = fnirs_file[patient] # (6act, 3hb, 3px, 4band, 6ch)
            for act in range(6):
                for h in range(3):
                    for p in range(3):
                        for band in range(4):
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

    zero_cols = []
    for col in all_cols:
        if sum(df[col]) == 0: zero_cols.append(col)
    print(zero_cols)
    df = df.drop(columns=zero_cols)
    print(df) # 4752

    if not os.path.exists('./csv'):
        os.makedirs('./csv')
    df.to_csv('./csv/' + save_csv_name + '.csv', sep=',')
    print(save_csv_name, 'saved.')


if __name__ == "__main__":

    full = False

    if full:
        sliced2csv(load_pth='./power_npy/full_sliced/', save_csv_name='only_eeg_power_sliced_full')
    else:
        # sliced2csv(load_pth='./power_npy/sliced/', save_csv_name='only_eeg_power_sliced')
        sliced2csv(load_pth='./power_npy/sliced/', save_csv_name='eeg_fnirs_power_sliced')
    

import numpy as np
import pandas as pd
import os

def Method1():

    eeg_root_path = './Experiments/Resting/EEG/new_scale/input_normalized/Method1/'
    fnirs_root_path = './Experiments/Resting/FNIRS/new_scale/input_normalized/Method1/'
    levels = ['AD', 'NORMAL', 'MCI']

    ###########################################################################

    # EEG
    eeg_ad_input = np.load(eeg_root_path + 'AD_input.npy') # (27, 640)
    eeg_norm_input = np.load(eeg_root_path + 'NORMAL_input.npy') # (70, 640)
    eeg_mci_input = np.load(eeg_root_path + 'MCI_input.npy') # (48, 640)

    # # Drop Entropy
    # entropy_col_idx = [4+5*i for i in range(160)]
    # print(entropy_col_idx)
    # eeg_ad_input = np.delete(eeg_ad_input, entropy_col_idx, 1)
    # eeg_norm_input = np.delete(eeg_norm_input, entropy_col_idx, 1)
    # eeg_mci_input = np.delete(eeg_mci_input, entropy_col_idx, 1)

    # action = 'Resting'
    # save_root = './Experiments/' + action + '/EEG/new_scale/input_normalized/Method1/'
    # np.save(save_root + 'AD_input.npy', eeg_ad_input)
    # np.save(save_root + 'NORMAL_input.npy', eeg_norm_input)
    # np.save(save_root + 'MCI_input.npy', eeg_mci_input)

    print(eeg_ad_input.shape) # (27, 640)
    print(eeg_norm_input.shape) # (70, 640)
    print(eeg_mci_input.shape) # (48, 640)

    eeg_bands = {'Delta': (1, 4),
                    'Theta': (4, 8),
                    'Alpha': (8, 13),
                    'Beta': (13, 30),
                    'Gamma': (30, 70)} # (70 -> 40)
    band_keys = list(eeg_bands.keys())

    # 800 = 5 feature * 5 band * 32 channel
    eeg_cols = []
    features = ['mu', 'std', 'skew', 'kurt']
    eeg_channels = 32
    for ch in range(eeg_channels): # 32
        for band in range(len(band_keys)): # 5
            for feat in range(len(features)): # 4
                col_name = 'eeg-ch-' + str(ch) + '-' + band_keys[band] + '-' + features[feat]
                eeg_cols.append(col_name)


    # print(eeg_cols)
    print(len(eeg_cols)) # 640

    ###########################################################################

    # FNIRS
    hb_fnirs_ad_input = np.load(fnirs_root_path + 'AD_input_Hb.npy') # (27, 96)
    hb_fnirs_norm_input = np.load(fnirs_root_path + 'NORMAL_input_Hb.npy') # (70, 96)
    hb_fnirs_mci_input = np.load(fnirs_root_path + 'MCI_input_Hb.npy') # (48, 96)

    hbo_fnirs_ad_input = np.load(fnirs_root_path + 'AD_input_Hb.npy') # (27, 96)
    hbo_fnirs_norm_input = np.load(fnirs_root_path + 'NORMAL_input_Hb.npy') # (70, 96)
    hbo_fnirs_mci_input = np.load(fnirs_root_path + 'MCI_input_Hb.npy') # (48, 96)

    # # Drop Entropy
    # entropy_col_idx = [4+5*i for i in range(24)]
    # print(entropy_col_idx)
    # hb_fnirs_ad_input = np.delete(hb_fnirs_ad_input, entropy_col_idx, 1)
    # hb_fnirs_norm_input = np.delete(hb_fnirs_norm_input, entropy_col_idx, 1)
    # hb_fnirs_mci_input = np.delete(hb_fnirs_mci_input, entropy_col_idx, 1)

    # hbo_fnirs_ad_input = np.delete(hbo_fnirs_ad_input, entropy_col_idx, 1)
    # hbo_fnirs_norm_input = np.delete(hbo_fnirs_norm_input, entropy_col_idx, 1)
    # hbo_fnirs_mci_input = np.delete(hbo_fnirs_mci_input, entropy_col_idx, 1)

    print(hb_fnirs_ad_input.shape) # (27, 96)
    print(hb_fnirs_norm_input.shape) # (70, 96)
    print(hb_fnirs_mci_input.shape) # (48, 96)

    print(hbo_fnirs_ad_input.shape) # (27, 96)
    print(hbo_fnirs_norm_input.shape) # (70, 96)
    print(hbo_fnirs_mci_input.shape) # (48, 96)

    # action = 'Resting'
    # save_root = './Experiments/' + action + '/FNIRS/new_scale/input_normalized/Method1/'
    # np.save(save_root + 'AD_input_Hb.npy', hb_fnirs_ad_input)
    # np.save(save_root + 'NORMAL_input_Hb.npy', hb_fnirs_norm_input)
    # np.save(save_root + 'MCI_input_Hb.npy', hb_fnirs_mci_input)

    # np.save(save_root + 'AD_input_HbO.npy', hbo_fnirs_ad_input)
    # np.save(save_root + 'NORMAL_input_HbO.npy', hbo_fnirs_norm_input)
    # np.save(save_root + 'MCI_input_HbO.npy', hbo_fnirs_mci_input)


    fnirs_band = {
                    'VLFO': (0.01, 0.04),
                    'LFO': (0.04, 0.15),
                    'Respiration': (0.16, 0.6),
                    'Heartbeat': (0.8, 1.5)
                }
    band_keys = list(fnirs_band.keys())

    # 120 = 5 feature * 4 band * 6 channel
    hb_fnirs_cols, hbo_fnirs_cols = [], []
    features = ['mu', 'std', 'skew', 'kurt']
    fnirs_channels = 6
    for ch in range(fnirs_channels): # 6
        for band in range(len(band_keys)): # 4
            for feat in range(len(features)): # 4
                col_name = 'fnirs-ch-' + str(ch) + '-' + band_keys[band] + '-' + features[feat]
                hb_fnirs_cols.append('hb-' + col_name)
                hbo_fnirs_cols.append('hbo-' + col_name)


    print(len(hb_fnirs_cols)) # 96
    print(len(hbo_fnirs_cols)) # 96
    fnirs_cols = hb_fnirs_cols + hbo_fnirs_cols

    ###########################################################################
    
    # 1) EEG + FNIRS
    # all_ad_input = np.concatenate([eeg_ad_input, hb_fnirs_ad_input, hbo_fnirs_ad_input], axis=-1)
    # all_norm_input = np.concatenate([eeg_norm_input, hb_fnirs_norm_input, hbo_fnirs_norm_input], axis=-1)
    # all_mci_input = np.concatenate([eeg_mci_input, hb_fnirs_mci_input, hbo_fnirs_mci_input], axis=-1)

    # # print(all_ad_input.shape, all_norm_input.shape, all_mci_input.shape) # 800 + 120 + 120 = 1040

    # all_input = np.concatenate([all_ad_input, all_norm_input, all_mci_input], axis=0)
    # labels = [0] * len(all_ad_input) + [1] * len(all_norm_input) + [2] * len(all_mci_input)
    # all_cols = eeg_cols + fnirs_cols

    # print(all_input.shape) # (145, 1040)
    # print(len(labels), len(all_cols)) # 145, 1040

    # df = pd.DataFrame(all_input, columns=all_cols)
    # df.insert(0, 'label', labels)
    # print(df)

    # save_csv_name = 'new_scale_normalized_method1_cwt'
    # if not os.path.exists('./csv'):
    #     os.makedirs('./csv')
    # df.to_csv('./csv/' + save_csv_name + '.csv', sep=',')
    # print(save_csv_name, 'saved.')
    
    ###########################################################################

    # # 2) only EEG
    # all_input = np.concatenate([eeg_ad_input, eeg_norm_input, eeg_mci_input], axis=0)
    # labels = [0] * len(eeg_ad_input) + [1] * len(eeg_norm_input) + [2] * len(eeg_mci_input)

    # print(all_input.shape) # (145, 640)
    # print(len(labels), len(eeg_cols)) # 145, 640

    # df = pd.DataFrame(all_input, columns=eeg_cols)
    # df.insert(0, 'label', labels)
    # print(df)

    # save_csv_name = 'only_eeg_new_scale_normalized_method1_cwt'
    # if not os.path.exists('./csv'):
    #     os.makedirs('./csv')
    # df.to_csv('./csv/' + save_csv_name + '.csv', sep=',')
    # print(save_csv_name, 'saved.')

    ###########################################################################

    # 3) only FNIRS
    all_ad_input = np.concatenate([hb_fnirs_ad_input, hbo_fnirs_ad_input], axis=-1)
    all_norm_input = np.concatenate([hb_fnirs_norm_input, hbo_fnirs_norm_input], axis=-1)
    all_mci_input = np.concatenate([hb_fnirs_mci_input, hbo_fnirs_mci_input], axis=-1)
    all_input = np.concatenate([all_ad_input, all_norm_input, all_mci_input], axis=0)

    labels = [0] * len(all_ad_input) + [1] * len(all_norm_input) + [2] * len(all_mci_input)

    print(all_input.shape) # (145, 192)
    print(len(labels), len(fnirs_cols)) # 145, 192

    df = pd.DataFrame(all_input, columns=fnirs_cols)
    df.insert(0, 'label', labels)
    print(df)

    save_csv_name = 'only_fnirs_new_scale_normalized_method1_cwt'
    if not os.path.exists('./csv'):
        os.makedirs('./csv')
    df.to_csv('./csv/' + save_csv_name + '.csv', sep=',')
    print(save_csv_name, 'saved.')


    

if __name__ == "__main__":

    Method1()

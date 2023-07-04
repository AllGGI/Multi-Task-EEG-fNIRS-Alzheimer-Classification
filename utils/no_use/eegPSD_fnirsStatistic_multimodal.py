import pandas as pd
import numpy as np
import os



def multiple_acts(total_X, total_y, act_list=[], norm=False): # 4752 / 6 -> 792

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

    return ret


if __name__ == "__main__":

    act_num = [0,1,2,3,4,5] # Resting=0, C1=1, C2=2, N1=3, N2=4, V=5

    slice_sec = 5
    open_csv = './final_csv/' + str(slice_sec) + 'sec_eeg_fnirs_power_sliced.csv'
    save_csv_name = str(slice_sec) + 'sec_eeg_PSD_fnirs_TimeDomain'
    
    # open_csv = './final_csv/full_eeg_fnirs_psd.csv'
    

    df = pd.read_csv(open_csv)
    df = df.drop('Unnamed: 0', axis=1)
    # df = df[np.isfinite(df).all(1)]

    X_total = df.drop('label', axis=1)
    y = df['label']

    # eeg PSD
    data_dict = multiple_acts(X_total, y, act_list=act_num, norm=True)
    X_eeg = data_dict['X_eeg']

    ####################################################################################
    # Time domain Fnirs
    time_domain_csv = './final_csv/fnirs_time_domain_features.csv'
    time_domain_fnirs = pd.read_csv(time_domain_csv)
    time_domain_fnirs = time_domain_fnirs.drop('Unnamed: 0', axis=1)
    # time_domain_fnirs = time_domain_fnirs[np.isfinite(time_domain_fnirs).all(1)]

    X_total_time = time_domain_fnirs.drop('label', axis=1)
    y_total_time = df['label'] # Same as 'y_total'

    temp_dict = multiple_acts(X_total_time, y_total_time, act_list=act_num, norm=True)
    X_fnirs = temp_dict['X_fnirs']

    # # remove problematic eeg row (= 72)
    # print(X_eeg[X_eeg.isna().any(axis=1)])
    # print('*'*100)
    # print(X_fnirs)
    # print('*'*100)
    # X_fnirs.drop([72], axis=0, inplace=True)
    # print(X_fnirs)

    assert len(X_eeg) == len(X_fnirs)

    eeg_psd_fnirs_fft_df = pd.concat([X_eeg, X_fnirs], axis=1)
    eeg_psd_fnirs_fft_df.insert(0, 'label', y)

    print('1) concat')
    print(eeg_psd_fnirs_fft_df)
    print('*'*100)
    # print(eeg_psd_fnirs_fft_df[eeg_psd_fnirs_fft_df.isna().any(axis=1)]) # 72
    eeg_psd_fnirs_fft_df = eeg_psd_fnirs_fft_df.dropna(axis=0)
    print('2) remove NA')
    print(eeg_psd_fnirs_fft_df)
    print('*'*100)

    if not os.path.exists('./final_csv'):
        os.makedirs('./final_csv')
    
    # EEG-PSD + FNIRS-FFT = 3456 + 864 + label 1
    eeg_psd_fnirs_fft_df.to_csv('./final_csv/' + save_csv_name + '.csv', sep=',')
    print(save_csv_name + '.csv saved.')



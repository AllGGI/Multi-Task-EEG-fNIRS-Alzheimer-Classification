import pandas as pd
import numpy as np
import os

if __name__ == "__main__":

    root_pth = './csv/'
    if not os.path.exists(root_pth):
        os.makedirs(root_pth)

    # Selected EEG & fNIRs features
    eeg_features_file = './selected_features/RFECV_SKF_eeg_5.txt'
    fnirs_features_file = './selected_features/RFECV_SKF_fnirs_5.txt'

    eeg_features, fnirs_features = [], []
    with open(eeg_features_file, 'r') as eeg:
        lines = eeg.readlines()
        for line_idx in range(3, len(lines)):
            feat = lines[line_idx].replace('\n', '')
            eeg_features.append(feat)
    with open(fnirs_features_file, 'r') as fnirs:
        lines = fnirs.readlines()
        for line_idx in range(3, len(lines)):
            feat = lines[line_idx].replace('\n', '')
            fnirs_features.append(feat)
    
    print(len(eeg_features)) # 10
    print(len(fnirs_features)) # 16

    eeg_df = pd.read_csv(root_pth + 'only_eeg_new_scale_normalized_method1_cwt.csv') # 193
    y_eeg = eeg_df['label']

    # print(eeg_df[eeg_features])
    X_eeg = eeg_df[eeg_features].drop('Unnamed: 0', axis=1)
    X_eeg.insert(0, 'label', y_eeg)
    # print(X_eeg)

    fnirs_df = pd.read_csv(root_pth + 'only_fnirs_new_scale_normalized_method1_cwt.csv') # 193
    # df = pd.DataFrame(X_eeg, columns=X_eeg.columns)
    # df.to_csv(root_pth + 'selected_eeg.csv', sep=',')
    
    # Save fNIRs.csv
    X_fnirs = fnirs_df[fnirs_features].drop('Unnamed: 0', axis=1)
    # print(X_fnirs)

    X_hybrid = pd.concat([X_eeg, X_fnirs], axis=1)
    # print(X_hybrid)

    X_hybrid = X_hybrid[np.isfinite(X_hybrid).all(1)]
    # print(X_hybrid)

    df = pd.DataFrame(X_hybrid, columns=X_hybrid.columns)
    df.to_csv(root_pth + 'selected_hybrid_new_scale_norm_hybrid_method1.csv', sep=',')


    

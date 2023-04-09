import pandas as pd
import numpy as np
import os

if __name__ == "__main__":

    include_np = False
    ishybrid = True

    df = pd.read_csv('./csv/eeg_fnirs_power_sliced_meansumstd.csv')
    df = df[np.isfinite(df).all(1)] # remove nans, 191


    X = df.drop('label', axis=1)
    y = df['label']

    if not ishybrid:
        # Selected EEG & fNIRs features
        eeg_features_file = './ETC_esti1000/RFECV_SKF_eeg_10.txt'
        fnirs_features_file = './ETC_esti1000/RFECV_SKF_fnirs_10.txt'
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
        
        eeg_fnirs_features = eeg_features + fnirs_features

        print(len(eeg_features))
        print(len(fnirs_features))
        print(len(eeg_fnirs_features))

        # Save EEG.csv
        X_eeg = X[eeg_features]
        X_eeg.insert(0, 'label', y)
        if not os.path.exists('./csv'):
            os.makedirs('./csv')
        df = pd.DataFrame(X_eeg, columns=X_eeg.columns)
        df.to_csv('./csv/selected_eeg.csv', sep=',')
        
        # Save fNIRs.csv
        X_fnirs = X[fnirs_features]
        X_fnirs.insert(0, 'label', y)
        if not os.path.exists('./csv'):
            os.makedirs('./csv')
        df = pd.DataFrame(X_fnirs, columns=X_fnirs.columns)
        df.to_csv('./csv/selected_fnirs.csv', sep=',')

        # # Save eeg_fnirs.csv
        # X_eegfnirs = X[eeg_fnirs_features]
        # X_eegfnirs.insert(0, 'label', y)
        # if not os.path.exists('./csv'):
        #     os.makedirs('./csv')
        # df = pd.DataFrame(X_eegfnirs, columns=X_eegfnirs.columns)
        # df.to_csv('./csv/selected_eegfnirs.csv', sep=',')

    
    else:
        hybrid_features_file = './rfecv_plots_200_act-1/RFECV_SKF_hybrid_10.txt'
        hybrid_features = []
        with open(hybrid_features_file, 'r') as hybrid:
            lines = hybrid.readlines()
            for line_idx in range(3, len(lines)):
                feat = lines[line_idx].replace('\n', '')
                hybrid_features.append(feat)

        print(len(hybrid_features))

        # Save hybrid.csv
        X_hybrid = X[hybrid_features]
        X_hybrid.insert(0, 'label', y)
        if not os.path.exists('./csv'):
            os.makedirs('./csv')
        df = pd.DataFrame(X_hybrid, columns=X_hybrid.columns)
        df.to_csv('./csv/power_sliced_meansumstd_hybrid_200-C1.csv', sep=',')


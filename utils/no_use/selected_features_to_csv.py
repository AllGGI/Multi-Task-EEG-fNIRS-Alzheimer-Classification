import pandas as pd
import numpy as np
import os

if __name__ == "__main__":

    to_csv_type = 'hybrid'

    if to_csv_type != 'hybrid':

        act_num = [0,1,2,3,4,5]
        act_types = {0:'R', 1:'C1', 2:'C2', 3:'N1', 4:'N2', 5:'V'}

        exp_type = 'ablation2/'
        csv_root = './final_csv/' + exp_type
        optimal_features_root = './FINAL_results/' + exp_type

        # pth = '5secEEGPSD_FullFnirsPSD_FullFnirsTimeDomain'
        # pth = '5sec_eeg_PSD_fnirs_TimeDomain'
        # pth = 'full_eeg_fnirs_psd'
        # pth = '3sec_eeg_fnirs_power_sliced'

        # pth = 'opt-2_5secEEGPSD_FullFnirsPSD'
        # pth = 'opt-3_5secEEGPSD_FullFnirsTimeDomain'
        pth = 'opt-4_FullFnirsPSD_FullFnirsTimeDomain'

        df = pd.read_csv(csv_root + pth + '.csv')
        df = df[np.isfinite(df).all(1)] # remove nans, 191
        X = df.drop('label', axis=1)
        y = df['label']

        fold_name = optimal_features_root + pth + '_' + '-'.join([str(i) for i in act_num]) + '/'


        if to_csv_type == 'eeg':
            eeg_features_file = fold_name + 'RFECV_SKF_eeg_10.txt'
            eeg_features = []
            with open(eeg_features_file, 'r') as eeg:
                lines = eeg.readlines()
                for line_idx in range(3, len(lines)):
                    feat = lines[line_idx].replace('\n', '')
                    eeg_features.append(feat)

            print(len(eeg_features))

            # Save EEG.csv
            X_eeg = X[eeg_features]
            X_eeg.insert(0, 'label', y)
            if not os.path.exists('./final_csv'):
                os.makedirs('./final_')
            df = pd.DataFrame(X_eeg, columns=X_eeg.columns)
            df.to_csv('./final_csv/RFECV-ONLY-EEG-' + pth + '_' + '-'.join([act_types[i] for i in act_num]) + '.csv', sep=',')

        elif to_csv_type == 'fnirs':
            fnirs_features_file = fold_name + 'RFECV_SKF_fnirs_10.txt'
            fnirs_features = []
            with open(fnirs_features_file, 'r') as fnirs:
                lines = fnirs.readlines()
                for line_idx in range(3, len(lines)):
                    feat = lines[line_idx].replace('\n', '')
                    fnirs_features.append(feat)

            print(len(fnirs_features))
                
            # Save fNIRs.csv
            X_fnirs = X[fnirs_features]
            X_fnirs.insert(0, 'label', y)
            if not os.path.exists('./csv'):
                os.makedirs('./csv')
            df = pd.DataFrame(X_fnirs, columns=X_fnirs.columns)
            df.to_csv('./final_csv/RFECV-ONLY-fNIRS-' + pth + '_' + '-'.join([act_types[i] for i in act_num]) + '.csv', sep=',')


    
    else:

        act_num = [0,1,2,3,4,5]
        act_types = {0:'R', 1:'C1', 2:'C2', 3:'N1', 4:'N2', 5:'V'}
        exp_type = 'ablation2/'
        csv_root = './final_csv/' + exp_type
        optimal_features_root = './FINAL_results/' + exp_type

        # pth = '5secEEGPSD_FullFnirsPSD_FullFnirsTimeDomain'
        # pth = '5sec_eeg_PSD_fnirs_TimeDomain'
        # pth = '3secEEGPSD_FullFnirsPSD_FullFnirsTimeDomain'
        # pth = 'opt-2_5secEEGPSD_FullFnirsPSD'
        pth = 'opt-3_5secEEGPSD_FullFnirsTimeDomain'


        df = pd.read_csv(csv_root + pth + '.csv')
        df = df[np.isfinite(df).all(1)] # remove nans, 191
        X = df.drop('label', axis=1)
        y = df['label']


        fold_name = optimal_features_root + pth + '_' + '-'.join([str(i) for i in act_num]) + '/'
        hybrid_features_file = fold_name + 'RFECV_SKF_hybrid_10.txt'
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
        if not os.path.exists('./final_csv'):
            os.makedirs('./final_csv')
        df = pd.DataFrame(X_hybrid, columns=X_hybrid.columns)
        df.to_csv('./final_csv/RFECV-' + pth + '_' + '-'.join([act_types[i] for i in act_num]) + '.csv', sep=',')

        print('./final_csv/RFECV-' + pth + '_' + '-'.join([act_types[i] for i in act_num]) + '.csv')

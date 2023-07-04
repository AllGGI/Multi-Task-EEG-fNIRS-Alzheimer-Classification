import pickle
import os
import numpy as np
import spkit as sp
from scipy.stats import pearsonr
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import matplotlib.pyplot as plt

class Comparison1():

    '''
        1. Paper: "An EEG-fNIRS hybridization technique in the four-class classification of alzheimer's disease"
        2. Task: memorize -> retrieval a number sequence
        3. Feature extraction
            0) Block average? => Average all trials of each tasks (= diff sequences)
            1) EEG (total 10 sec)
                - 2 second time window * 5 segment (no overlap)
                - for each time window, for each band:
                    Feature = Relative Band Power = Each band power / Sum(all band Power)
                    (Band power = PSD w/ FFT)
                - 33 channel * 6 band * 5 time window * 4 block = 3960 features
            2) fNIRS (3s ~ 12s = total 9 sec)
                - 3 second time window * 3 segment (no overlap)
                - Feature = Avg changes of HbO, HbR concentrations
                - 46 channel * 2 Hb * 3 time window * 4 block = 1104 features
        4. Feature selection
            - PCCFS = Pearson correalation coefficient-based feature selection
            - optimal EEG + optimal fNIRS -> optimal hybrid features
        5. Classifier
            - LOOCV + LDA
            (Leave One Out Cross Validation + Linear Discriminant Analysis)
    
    '''
    def __init__(self, slice_sec_num=10):

        self.slice_sec_num = slice_sec_num
        self.eeg_list, self.fnirs_list = [], []
        self.levels = ['AD', 'NORMAL', 'MCI']

        self.eeg_npy, self.fnirs_npy = {}, {}
        self.loaded_df = None
        self.output_pth = './previous_study_implementation/Cicalese_et_al/outputs/'
        self.input_pth = './previous_study_implementation/Cicalese_et_al/inputs/'

        if not os.path.exists(self.output_pth):
            os.makedirs(self.output_pth)
        if not os.path.exists(self.input_pth):
            os.makedirs(self.input_pth)

        self.fnirs_window_num = 3

    def load_pickle(self):
        sliced_pth = './pickles/eeg-' + str(self.slice_sec_num) + 'sec_fnirs-full_baselinecorrected_fnirs_sliced.pkl'
        new_AD_sliced_pth = './pickles/new-AD_eeg-' + str(self.slice_sec_num) + 'sec_fnirs-full_baselinecorrected_fnirs_sliced.pkl'

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
        self.eeg_list = [ad_eeg, norm_eeg, mci_eeg]
        self.fnirs_list = [ad_fnirs, norm_fnirs, mci_fnirs]

    def comparison1_eeg(self):
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

        for level_idx, level_data in enumerate(self.eeg_list): # ad 26 -> norm 64 -> mci 46
            patients = []
            for patient in level_data: # 26
                act_out = []
                for act_num, act_data in enumerate(patient): # 6 = RO, C1, C2, N1, N2, V
                    if act_num == 3 or act_num == 4: # C, N (1500=[-1,2 sec], 32)
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
                        act_out.append(this_act_out) # (32, 6, 5)

                act_out = np.stack(act_out, axis=-1) # (32ch, 6 band, 5 window, 2 block)
                patients.append(act_out)
            
            patients = np.stack(patients, axis=0)
            print(self.levels[level_idx], patients.shape) # AD (35, 32, 6, 5, 2)

            np.save(self.input_pth + self.levels[level_idx] + '_eeg_RBP.npy', patients)
            print('>> npy saved.')

    def comparison1_fnirs(self):
        '''
        3. Feature extraction
            0) Block average? => Average all trials of each tasks (= diff sequences)
            2) fNIRS (3s ~ 12s = total 9 sec)
                - 3 second time window * 3 segment (no overlap)
                - Feature = Avg changes of HbO, HbR concentrations
                (By taking avg of data every 3 s with no overlapping for block averaged signals.)
                - 46 channel * 2 Hb * 3 time window * 4 block = 1104 features
        
        '''

        fnirs_sampling_rate = 8
        fnirs_channels = 6
        for level_idx, level_data in enumerate(self.fnirs_list): # ad 26 -> norm 64 -> mci 46
            patients = []
            for patient_idx, patient in enumerate(level_data): # 26
                act_out = []
                for act_num, act_data in enumerate(patient): # 6 = RO, C1, C2, N1, N2, V
                    if act_num == 3 or act_num == 4: # (80, 6)
                        hb_data_list = act_data['Hb']
                        hbo_data_list = act_data['HbO']

                        hb_data_list = [hb[:,:] for hb in hb_data_list] # 1sec~10sec = 9s
                        hbo_data_list = [hbo[:,:] for hbo in hbo_data_list] # 1sec~10sec = 9s
                        
                        hb_data_list = np.stack(hb_data_list, axis=0)
                        hb_block_avg = np.mean(hb_data_list, axis=0)
                        hb_time_windows = [np.mean(hb_block_avg[24*i:24*(i+1), :], axis=0) for i in range(self.fnirs_window_num)] # 5 sec * 2 window
                        hb_time_windows = np.stack(hb_time_windows, axis=-1) # (6 ch, 2 windows)

                        hbo_data_list = np.stack(hbo_data_list, axis=0)
                        hbo_block_avg = np.mean(hbo_data_list, axis=0)
                        hbo_time_windows = [np.mean(hbo_block_avg[24*i:24*(i+1), :], axis=0) for i in range(self.fnirs_window_num)] # 5 sec * 2 window
                        hbo_time_windows = np.stack(hbo_time_windows, axis=-1) # (6 ch, 2 windows)

                        this_act_out = np.stack([hb_time_windows, hbo_time_windows], axis=1)
                        act_out.append(this_act_out)

                act_out = np.stack(act_out, axis=-1) # (6ch, 2 hb, 3 window, 2 block)
                patients.append(act_out)
            
            patients = np.stack(patients, axis=0)
            print(self.levels[level_idx], patients.shape) # AD (35, 6, 2, 3, 2)

            np.save(self.input_pth + self.levels[level_idx] + '_fnirs_avg.npy', patients)
            print('>> npy saved.')

    def data2csv(self):
        for level in self.levels:
            self.eeg_npy[level] = np.load(self.input_pth + level + '_eeg_RBP.npy')
            self.fnirs_npy[level] = np.load(self.input_pth + level + '_fnirs_avg.npy')

        # EEG: (35 patients, 32 ch, 6 band, 5 window, 2 block)
        eeg_cols = []
        eeg_bands = ['Delta', 'Theta', 'LowAlpha', 'HighAlpha', 'Beta', 'Gamma']
        for ch in range(32):
            for band in range(len(eeg_bands)):
                for window in range(5):
                    for block in range(2):
                        col_name = 'eeg_ch-' + str(ch) + '_band-' + eeg_bands[band] + '_window-' + str(window) + '_block-' + str(block)
                        eeg_cols.append(col_name)

        print(len(eeg_cols)) # 1920

        # fNIRS: (35 patients, 6 ch, 2 hb, 2 window, 2 block)
        fnirs_cols = []
        fnirs_hb = ['Hb', 'HbO']
        for ch in range(6):
            for h in range(len(fnirs_hb)):
                for window in range(self.fnirs_window_num):
                    for block in range(2):
                        col_name = 'fnirs_ch-' + str(ch) + '_' + fnirs_hb[h] + '_window-' + str(window) + '_block-' + str(block)
                        fnirs_cols.append(col_name)

        print(len(fnirs_cols)) # 72

        ################################################################################
        # EEG
        EEG_all_levels = []
        for level in self.levels:
            level_flatten_data = []
            for patient in range(self.eeg_npy[level].shape[0]):
                flatten_data = []
                this_patient_data = self.eeg_npy[level][patient]
                for ch in range(32):
                    for band in range(len(eeg_bands)):
                        for window in range(5):
                            for block in range(2):
                                flatten_data.append(this_patient_data[ch][band][window][block])
                
                level_flatten_data.append(flatten_data)

            level_flatten_data = np.asarray(level_flatten_data)
            print(level_flatten_data.shape)
            EEG_all_levels.append(level_flatten_data)

        ################################################################################
        # FNIRS
        FNIRS_all_levels = []
        for level in self.levels:
            level_flatten_data = []
            for patient in range(self.fnirs_npy[level].shape[0]):
                flatten_data = []
                this_patient_data = self.fnirs_npy[level][patient]
                for ch in range(6):
                    for h in range(len(fnirs_hb)):
                        for window in range(self.fnirs_window_num):
                            for block in range(2):
                                flatten_data.append(this_patient_data[ch][h][window][block])
                
                level_flatten_data.append(flatten_data)

            level_flatten_data = np.asarray(level_flatten_data)
            print(level_flatten_data.shape) # (26, 1296)
            FNIRS_all_levels.append(level_flatten_data)

        all_levels = [np.concatenate([EEG_all_levels[level], FNIRS_all_levels[level]], axis=1) for level in range(len(self.levels))]
        print(all_levels[0].shape, all_levels[1].shape, all_levels[2].shape) # (?, 1920+72=1992)
        

        all_input = np.concatenate(all_levels, axis=0) # (35+64+46=144, 1992)
        labels = [0] * len(all_levels[0]) + [1] * len(all_levels[1]) + [2] * len(all_levels[2])
        all_cols = eeg_cols + fnirs_cols

        assert all_input.shape[0] == len(labels) and all_input.shape[1] == len(all_cols)

        df = pd.DataFrame(all_input, columns=all_cols)
        df.insert(0, 'label', labels)
        print(df) # (145 x 1993)
        print('*'*100)

        df = df[np.isfinite(df).all(1)]
        if not os.path.exists('./previous_study_implementation/'):
            os.makedirs('./previous_study_implementation/')
        df.to_csv('./previous_study_implementation/comparison1_eegfnirs_init.csv', sep=',')
        print('./previous_study_implementation/comparison1_eegfnirs_init.csv saved.')

    def find_optimal_features(self, X_source, y_source, init_feat):

        max_acc = 0.0
        optimal_feat = []
        for feat in tqdm(init_feat):
            optimal_feat.append(feat)
            this_X = X_source[optimal_feat]
            
            cv = LeaveOneOut()
            model = LinearDiscriminantAnalysis()

            scores = cross_val_score(model, this_X, y_source, scoring='accuracy', cv=cv, n_jobs=-1)
            mean_score = round(np.mean(scores), 4)
            # print(len(scores)) # 144
            # print(mean_score) # 0.4167
            if mean_score > max_acc:
                max_acc = mean_score # and left feat in optimal list
            else:
                optimal_feat.remove(feat) # if feat lower the acc, remove this from optimal list

        return optimal_feat, max_acc

    def load_csv(self):
        self.loaded_df = pd.read_csv('./previous_study_implementation/Cicalese_et_al/comparison1_eegfnirs_init.csv')

        loaded_df = self.loaded_df.drop('Unnamed: 0', axis=1)
        loaded_df = loaded_df[np.isfinite(loaded_df).all(1)]
        loaded_df = loaded_df.reset_index(drop=True)

        return loaded_df


    def feature_selection(self):

        df = self.load_csv()
        X_total = df.drop('label', axis=1)
        y = df['label']

        hybrid_features = X_total.columns.tolist()
        # print(features)
        # print(len(features)) # 1992

        fnirs_start_idx = -1
        for idx, col in enumerate(hybrid_features):
            if 'fnirs' in col:
                fnirs_start_idx = idx
                break
        
        eeg_features = hybrid_features[:fnirs_start_idx] # 1920
        fnirs_features = hybrid_features[fnirs_start_idx:] # 72

        ############################ PCCFS ###################################
        # EEG feature sorting
        eeg_correlations = {}
        for feature in eeg_features:
            coef = pearsonr(df[feature], y).statistic
            if not np.isnan(coef): eeg_correlations[feature] = coef
        print(len(eeg_correlations)) # 1920
        sorted_eeg_features = sorted(eeg_correlations.items(), key=lambda x: abs(x[1]), reverse=True)
        
        # fNIRS feature sorting
        fnirs_correlations = {}
        for feature in fnirs_features:
            coef = pearsonr(df[feature], y).statistic
            if not np.isnan(coef): fnirs_correlations[feature] = coef
        print(len(fnirs_correlations))
        sorted_fnirs_features = sorted(fnirs_correlations.items(), key=lambda x: abs(x[1]), reverse=True)

        init_eeg_feat = []
        for feature, correlation in sorted_eeg_features:
            # print(f'Feature: {feature}, Correlation: {correlation}')
            init_eeg_feat.append(feature)
        init_fnirs_feat = []
        for feature, correlation in sorted_fnirs_features:
            # print(f'Feature: {feature}, Correlation: {correlation}')
            init_fnirs_feat.append(feature)

        ######################################################################
        # Feature selection
        optimal_eeg_feat, max_eeg_acc = self.find_optimal_features(X_total, y, init_feat=init_eeg_feat)
        optimal_fnirs_feat, max_fnirs_acc = self.find_optimal_features(X_total, y, init_feat=init_fnirs_feat)

        hybrid_features = optimal_eeg_feat + optimal_fnirs_feat
        # EEG+fNIRS hybrid feature sorting
        hybrid_correlations = {}
        for feature in hybrid_features:
            coef = pearsonr(df[feature], y).statistic
            if not np.isnan(coef): hybrid_correlations[feature] = coef
        print(len(hybrid_correlations))
        sorted_hybrid_features = sorted(hybrid_correlations.items(), key=lambda x: abs(x[1]), reverse=True)
        init_hybrid_feat = []
        for feature, correlation in sorted_hybrid_features:
            # print(f'Feature: {feature}, Correlation: {correlation}')
            init_hybrid_feat.append(feature)
        optimal_hybrid_feat, max_hybrid_acc = self.find_optimal_features(X_total, y, init_feat=init_hybrid_feat)

        print('>> Optimal eeg feat:', len(optimal_eeg_feat)) # 1920
        print('>> Optimal fnirs feat:', len(optimal_fnirs_feat)) # 72
        print('>> Optimal hybrid feat:', len(optimal_hybrid_feat))

        with open(self.output_pth + 'comparison1_optimal_eeg.txt', 'w') as f:
            f.write(">> Optimal number of features : %d \n" % len(optimal_eeg_feat))
            f.write(">> Max accuracy: " + str(max_eeg_acc) + '\n')
            f.write(">> Selected features:\n")
            for feat in optimal_eeg_feat:
                f.write(feat + '\n')
        with open(self.output_pth + 'comparison1_optimal_fnirs.txt', 'w') as f:
            f.write(">> Optimal number of features : %d \n" % len(optimal_fnirs_feat))
            f.write(">> Max accuracy: " + str(max_fnirs_acc) + '\n')
            f.write(">> Selected features:\n")
            for feat in optimal_fnirs_feat:
                f.write(feat + '\n')
        with open(self.output_pth + 'comparison1_optimal_hybrid.txt', 'w') as f:
            f.write(">> Optimal number of features : %d \n" % len(optimal_hybrid_feat))
            f.write(">> Max accuracy: " + str(max_hybrid_acc) + '\n')
            f.write(">> Selected features:\n")
            for feat in optimal_hybrid_feat:
                f.write(feat + '\n')

    def test(self):
        '''
            Test selected features with LDA
        '''
        # # EEG
        # optimal_eeg_feat = []
        # with open('./previous_study_implementation/comparison1_optimal_eeg.txt', 'r') as eeg:
        #     lines = eeg.readlines()
        #     for line_idx in range(3, len(lines)):
        #         feat = lines[line_idx].replace('\n', '')
        #         optimal_eeg_feat.append(feat)
        # # fNIRS
        # optimal_fnirs_feat = []
        # with open('./previous_study_implementation/comparison1_optimal_fnirs.txt', 'r') as fnirs:
        #     lines = fnirs.readlines()
        #     for line_idx in range(3, len(lines)):
        #         feat = lines[line_idx].replace('\n', '')
        #         optimal_fnirs_feat.append(feat)
        # Hybrid
        optimal_hybrid_feat = []
        with open(self.output_pth + 'comparison1_optimal_hybrid.txt', 'r') as hybrid:
            lines = hybrid.readlines()
            for line_idx in range(3, len(lines)):
                feat = lines[line_idx].replace('\n', '')
                optimal_hybrid_feat.append(feat)
        
        df = self.load_csv()
        selected_X = df.drop('label', axis=1)[optimal_hybrid_feat]
        label = df['label']

        cv = LeaveOneOut()
        model = LinearDiscriminantAnalysis()

        acc = cross_val_score(model, selected_X, label, scoring='accuracy', cv=cv)
        f1_acc = cross_val_score(model, selected_X, label, scoring='f1_macro', cv=cv)

        print('>> Mean Acc:', np.round(np.mean(acc), 4))
        print('>> Mean F1:', np.round(np.mean(f1_acc), 4))
        
    
    def run(self):
        '''
        3. Feature extraction (EEG, fNIRS)
        4. Feature selection
            - PCCFS = Pearson correalation coefficient-based feature selection
            - optimal EEG + optimal fNIRS -> optimal hybrid features
        5. Classification
            - LOOCV + LDA
            (Leave One Out Cross Validation + Linear Discriminant Analysis)
        
        '''
        # >> 3. Feature extraction (EEG, fNIRS)
        self.load_pickle()
        self.comparison1_eeg()
        self.comparison1_fnirs()
        self.data2csv()

        # >> 4. Feature selection & 5. Classification
        self.feature_selection()

        # >> 6. Test
        self.test()


if __name__ == "__main__":

    comp1 = Comparison1(slice_sec_num=10)
    comp1.run()



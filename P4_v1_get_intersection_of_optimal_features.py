import pandas as pd
import numpy as np


def get_intersection_of_folds(act_num, pth, cv_num, save_root):

    fold_dict = {}
    for fold_num in range(5):

        save_path = save_root + pth + '_' + '-'.join([str(i) for i in act_num]) + '/Fold_' + str(fold_num) + '/'
        # print('>> save path:', save_path)

        # Hybrid file
        # hybrid_features_file = save_path + 'RFECV_SKF_hybrid_' + str(cv_num) + '.txt'
        # hybrid_features = []
        # with open(hybrid_features_file, 'r') as hybrid:
        #     lines = hybrid.readlines()
        #     for line_idx in range(3, len(lines)):
        #         feat = lines[line_idx].replace('\n', '')
        #         hybrid_features.append(feat)

        # print(fold_num, len(hybrid_features))

        # EEG + fNIRS file
        eeg_features_file = save_path + 'RFECV_SKF_eeg_' + str(cv_num) + '.txt'
        fnirs_features_file = save_path + 'RFECV_SKF_fnirs_' + str(cv_num) + '.txt'

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
        
        # print('>> EEG:', len(eeg_features))
        # print('>> FNIRS:', len(fnirs_features))
        hybrid_features = eeg_features + fnirs_features


        fold_dict[fold_num] = set(hybrid_features)

    intersection_hybrid_features = list(fold_dict[0] & fold_dict[1] & fold_dict[2] & fold_dict[3] & fold_dict[4])
    print('\n>> Common hybrid features:', len(intersection_hybrid_features))
    # print(intersection_hybrid_features)

    # Save common features
    np.save(save_root + pth + '_' + '-'.join([str(i) for i in act_num]) + '/hybrid_all_fold.npy', intersection_hybrid_features)

    return intersection_hybrid_features


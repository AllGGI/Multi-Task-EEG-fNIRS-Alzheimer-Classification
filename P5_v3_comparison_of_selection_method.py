import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from tqdm import tqdm

from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import matplotlib.pyplot as plt

def find_optimal_features(X_source, y_source, init_feat):

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


def load_csv(csv_root):
    loaded_df = pd.read_csv(csv_root + 'comparison1_eegfnirs_allact.csv')
    

    loaded_df = loaded_df.drop('Unnamed: 0', axis=1)
    loaded_df = loaded_df[np.isfinite(loaded_df).all(1)]
    loaded_df = loaded_df.reset_index(drop=True)

    return loaded_df

def PrevStudy_feature_selection(csv_root):

    df = load_csv(csv_root)
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
    optimal_eeg_feat, max_eeg_acc = find_optimal_features(X_total, y, init_feat=init_eeg_feat)
    optimal_fnirs_feat, max_fnirs_acc = find_optimal_features(X_total, y, init_feat=init_fnirs_feat)

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
    optimal_hybrid_feat, max_hybrid_acc = find_optimal_features(X_total, y, init_feat=init_hybrid_feat)

    print('>> Optimal eeg feat:', len(optimal_eeg_feat)) # 1920
    print('>> Optimal fnirs feat:', len(optimal_fnirs_feat)) # 72
    print('>> Optimal hybrid feat:', len(optimal_hybrid_feat))

    with open('./Experimental_results/Experiment3/results/Prev_Extraction_Prev_Selection/comparison1_optimal_eeg.txt', 'w') as f:
        f.write(">> Optimal number of features : %d \n" % len(optimal_eeg_feat))
        f.write(">> Max accuracy: " + str(max_eeg_acc) + '\n')
        f.write(">> Selected features:\n")
        for feat in optimal_eeg_feat:
            f.write(feat + '\n')
    with open('./Experimental_results/Experiment3/results/Prev_Extraction_Prev_Selection/comparison1_optimal_fnirs.txt', 'w') as f:
        f.write(">> Optimal number of features : %d \n" % len(optimal_fnirs_feat))
        f.write(">> Max accuracy: " + str(max_fnirs_acc) + '\n')
        f.write(">> Selected features:\n")
        for feat in optimal_fnirs_feat:
            f.write(feat + '\n')
    with open('./Experimental_results/Experiment3/results/Prev_Extraction_Prev_Selection/comparison1_optimal_hybrid.txt', 'w') as f:
        f.write(">> Optimal number of features : %d \n" % len(optimal_hybrid_feat))
        f.write(">> Max accuracy: " + str(max_hybrid_acc) + '\n')
        f.write(">> Selected features:\n")
        for feat in optimal_hybrid_feat:
            f.write(feat + '\n')

def PrevStudy_test(csv_root, save_root):
    '''
        Test selected features with LDA
    '''
    # # EEG
    # optimal_eeg_feat = []
    # with open('./COMPARISON/comparison1_optimal_eeg.txt', 'r') as eeg:
    #     lines = eeg.readlines()
    #     for line_idx in range(3, len(lines)):
    #         feat = lines[line_idx].replace('\n', '')
    #         optimal_eeg_feat.append(feat)
    # # fNIRS
    # optimal_fnirs_feat = []
    # with open('./COMPARISON/comparison1_optimal_fnirs.txt', 'r') as fnirs:
    #     lines = fnirs.readlines()
    #     for line_idx in range(3, len(lines)):
    #         feat = lines[line_idx].replace('\n', '')
    #         optimal_fnirs_feat.append(feat)
    # Hybrid
    optimal_hybrid_feat = []
    with open(save_root + 'Prev_Extraction_Prev_Selection/comparison1_optimal_hybrid.txt', 'r') as hybrid:
        lines = hybrid.readlines()
        for line_idx in range(3, len(lines)):
            feat = lines[line_idx].replace('\n', '')
            optimal_hybrid_feat.append(feat)
    
    df = load_csv(csv_root)
    selected_X = df.drop('label', axis=1)[optimal_hybrid_feat]
    label = df['label']

    cv = LeaveOneOut()
    clf = LinearDiscriminantAnalysis()

    # acc = cross_val_score(model, selected_X, label, scoring='accuracy', cv=cv)
    # f1_acc = cross_val_score(model, selected_X, label, scoring='f1_macro', cv=cv)

    f1_list, acc_list = [], []
    proba_list = [] # To force AUC
    pred_list, gt_list = [], []
    total_confusion = np.zeros((3, 3))
    for i, (train_index, test_index) in enumerate(cv.split(selected_X)):

        X_train, X_test = selected_X.loc[train_index,:], selected_X.loc[test_index,:]
        y_train, y_test = label.loc[train_index], label.loc[test_index]

        # print(y_test.to_list().count(0), y_test.to_list().count(1), y_test.to_list().count(2))

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)

        acc_list.append(np.round(accuracy_score(y_test, y_pred), 4))
        f1_list.append(np.round(f1_score(y_test, y_pred, average='macro'), 4))

        proba_list.append(y_proba.squeeze())        
        pred_list.append(y_pred[0])
        gt_list.append(y_test.to_list()[0])


    proba_list = np.stack(proba_list, axis=0)
    # print(proba_list.shape, label.shape)
    print('>> Mean Acc:', np.round(np.mean(acc_list), 4))
    print('>> Mean F1:', np.round(np.mean(f1_list), 4))
    print('>> AUC:', np.round(roc_auc_score(label, proba_list, average='macro', multi_class='ovr'), 4))
    
    total_confusion = confusion_matrix(gt_list, pred_list)
    print('>> Confusion Matrix:')
    print(total_confusion)

    disp = ConfusionMatrixDisplay(total_confusion) # AD: 35 | CN: 63 | MCI: 46
    disp.plot()
    plt.show()






import numpy as np
import pandas as pd
# import seaborn as sns
import matplotlib.pyplot as plt
import os
from time import time
import datetime

# RFECV (Recursive Feature Elimination with Cross-Validation to select features)
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV 
from sklearn.datasets import make_classification
from sklearn.pipeline import Pipeline

# LOOCV + LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import LeaveOneOut

from sklearn.ensemble import ExtraTreesClassifier


def RFECV4AD(X, y, data_type, estimator, step, loo_lda, cv_num, min_features, save_pth):

    if loo_lda:
        cv = LeaveOneOut()
        cv_name = 'loo'
    else:
        cv = StratifiedKFold(cv_num)
        cv_name = 'SKF'

    rfecv = RFECV(
        estimator=estimator,
        step=step,
        cv=cv,
        scoring='f1_macro',
        min_features_to_select=min_features,
        n_jobs=-1 # use all cores
    )
    rfecv.fit(X, y)
    # rfecv_coef = np.absolute(rfecv.estimator_.coef_)
    print("Optimal number of features : %d" % rfecv.n_features_)
    selected_features = list(X.columns[rfecv.support_]) # 342

    # Plot number of features VS. cross-validation scores
    n_scores = len(rfecv.cv_results_["mean_test_score"]) # Mean of scores over the folds.
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Mean test accuracy")
    plt.errorbar(
        range(min_features, n_scores + min_features),
        rfecv.cv_results_["mean_test_score"],
        yerr=rfecv.cv_results_["std_test_score"],
    )
    plt.title("Recursive Feature Elimination \nwith correlated features")

    # plt.show()

    if not os.path.exists(save_pth):
        os.makedirs(save_pth)

    plt.savefig(save_pth + 'RFECV_' + cv_name + '_' + data_type + '_' + str(cv_num) + '.png')

    with open(save_pth + 'RFECV_' + cv_name + '_' + data_type + '_' + str(cv_num) + '.txt', 'w') as f:
        f.write(cv_name + '\n')
        f.write(">> Optimal number of features : %d \n" % rfecv.n_features_)
        f.write(">> Selected features:\n")
        for feat in selected_features:
            f.write(feat + '\n')


    return selected_features, max(rfecv.cv_results_["mean_test_score"])




def multiple_acts(total_X, total_y, act_list=[], norm=True): # 4752 / 6 -> 792

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

    print('>> Original features num (EEG, FNIRS):', len(eeg_features), len(fnirs_features))

    return ret



if __name__ == "__main__":

    # hybrid = False
    act_num = [0,1,2,3,4,5] # Resting=0, C1=1, C2=2, N1=3, N2=4, V=5

    pth = 'ablation1/5secEEGPSD_FullFnirsPSD_FullFnirsTimeDomain'
    # pth = 'ablation2/opt-2_5secEEGPSD_FullFnirsPSD'
    # pth = 'ablation2/opt-3_5secEEGPSD_FullFnirsTimeDomain'
    # pth = 'ablation2/opt-4_FullFnirsPSD_FullFnirsTimeDomain'
    # pth = 'ablation2/opt-5_5secEEGPSD'
    # pth = 'ablation2/opt-6_FullFnirsPSD'
    # pth = 'ablation2/opt-7_FullFnirsTimeDomain'


    # pth = '5sec_eeg_PSD_fnirs_TimeDomain'
    # pth = '3sec_eeg_fnirs_power_sliced'
    # pth = 'FullEEGPSD_FullFnirsPSD_FullFnirsTimeDomain'

    open_csv = './final_csv/'+ pth + '.csv'
    save_path='./FINAL_results/' + pth + '_' + '-'.join([str(i) for i in act_num]) + '/'
    print('>> save path:', save_path)


    df = pd.read_csv(open_csv)
    df = df.drop('Unnamed: 0', axis=1)

    print(df)
    df = df[np.isfinite(df).all(1)]
    print(df)

    X_total = df.drop('label', axis=1)
    y = df['label']

    ####################################################################
    # 1) all act
    # all_act(X_total, y)

    # 2) only 1 act
    data_dict = multiple_acts(X_total, y, act_list=act_num, norm=False)

    ####################################################################
    
    
    start_time = time()
    ####################################################################
    # 1) EEG / fNIRS each

    # RFECV
    cv_nums = [10]
    ret = {k:[] for k in cv_nums}

    '''
    # SVC
    # clf = SVC(kernel="linear") # estimator
    # ExtraTreesClassifier
    clf = ExtraTreesClassifier(n_estimators=100, random_state=2222, class_weight='balanced')

    
    for cv_num in cv_nums:
        eeg_selected_features, eeg_max_test_mean_score = RFECV4AD(X=data_dict['X_eeg'], y=data_dict['y_eeg'], data_type='eeg', estimator=clf, step=1,
                                        loo_lda=False, cv_num=cv_num, min_features=10, save_pth=save_path)
        fnirs_selected_features, fnirs_max_test_mean_score = RFECV4AD(X=data_dict['X_fnirs'], y=data_dict['y_fnirs'], data_type='fnirs', estimator=clf, step=1,
                                        loo_lda=False, cv_num=cv_num, min_features=10, save_pth=save_path)

        ret[cv_num].append(len(eeg_selected_features))
        ret[cv_num].append(eeg_max_test_mean_score)
        ret[cv_num].append(len(fnirs_selected_features))
        ret[cv_num].append(fnirs_max_test_mean_score)

    print(ret)
    '''

    ####################################################################
    # 2) EEG + fNIRS hybrid

    eeg_features_file = save_path + 'RFECV_SKF_eeg_10.txt'
    fnirs_features_file = save_path + 'RFECV_SKF_fnirs_10.txt'


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
    
    print('>> EEG:', len(eeg_features))
    print('>> FNIRS:', len(fnirs_features))
    hybrid_features = eeg_features + fnirs_features
    X_hybrid = X_total[hybrid_features]


    # RFECV
    cv_nums = [10]
    ret = {k:[] for k in cv_nums}


    # SVC
    # clf = SVC(kernel="linear") # estimator
    # ExtraTreesClassifier
    clf = ExtraTreesClassifier(n_estimators=200, random_state=2222, class_weight='balanced')

    for cv_num in cv_nums:
        hybrid_selected_features, max_mean_test_score = RFECV4AD(X=X_hybrid, y=y, data_type='hybrid', estimator=clf, step=1,
                                        loo_lda=False, cv_num=cv_num, min_features=30, save_pth=save_path)

        ret[cv_num].append(len(hybrid_selected_features))
        ret[cv_num].append(max_mean_test_score)

    print(ret)

    ####################################################################
        
    end_time = time()
    total_sec = end_time - start_time
    total_time = str(datetime.timedelta(seconds=total_sec)).split(".")
    print(">> Total time:", total_time[0], "\n")



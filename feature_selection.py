import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

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

def RFECV4AD(X, y, data_type, estimator, step, loo_lda, cv_num, min_features):

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
        scoring="accuracy",
        min_features_to_select=min_features,
    )
    rfecv.fit(X, y)
    # rfecv_coef = np.absolute(rfecv.estimator_.coef_)
    print("Optimal number of features : %d" % rfecv.n_features_)
    selected_features = list(X.columns[rfecv.support_]) # 342

    # # Plot number of features VS. cross-validation scores
    # plt.figure()
    # plt.xlabel("Number of features selected")
    # plt.ylabel("Cross validation score (accuracy)")
    # plt.plot(
    #     range(min_features, len(rfecv.grid_scores_)*step + min_features, step),
    #     rfecv.grid_scores_,
    # )

    # # plt.show()
    # plt.savefig('./RFECV_' + cv_name + '_' + data_type + '_' + str(cv_num) + '.png')

    save_path = './selected_features/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open(save_path + 'RFECV_' + cv_name + '_' + data_type + '_' + str(cv_num) + '.txt', 'w') as f:
        f.write(cv_name + '\n')
        f.write(">> Optimal number of features : %d \n" % rfecv.n_features_)
        f.write(">> Selected features:\n")
        for feat in selected_features:
            f.write(feat + '\n')

    return selected_features




if __name__ == "__main__":

    hybrid = False

    eeg_df = pd.read_csv('./csv/only_eeg_new_scale_normalized_method1_cwt.csv')
    eeg_df = eeg_df[np.isfinite(eeg_df).all(1)] # remove nans, 191
    X_eeg = eeg_df.drop('label', axis=1)
    y_eeg = eeg_df['label']

    fnirs_df = pd.read_csv('./csv/only_fnirs_new_scale_normalized_method1_cwt.csv')
    fnirs_df = fnirs_df[np.isfinite(eeg_df).all(1)] # remove nans, 191
    X_fnirs = fnirs_df.drop('label', axis=1)
    y_fnirs = fnirs_df['label']

    X_total = pd.concat([X_eeg, X_fnirs], axis=1)
    print(X_total)
    
    
    if not hybrid: # eeg / fnirs each

        # RFECV
        cv_nums = [5, 10]
        ret = {k:[] for k in cv_nums}

        # # LOOCV + LDA
        # # loo = LeaveOneOut()
        # lda = LinearDiscriminantAnalysis()
        # # print(cross_val_score(estimator=lda, X=X_eeg, y=y, cv=loo))

        # # RFECV4AD = RFECV for AD
        # eeg_selected_features = RFECV4AD(X=X_eeg, y=y, data_type='eeg', estimator=lda, step=1,
        #                                 loo_lda=True, cv_num=0, min_features=30)
        # fnirs_selected_features = RFECV4AD(X=X_fnirs, y=y, data_type='fnirs', estimator=lda, step=1,
        #                                 loo_lda=True, cv_num=0, min_features=10)

        # print(len(eeg_selected_features))
        # print(len(fnirs_selected_features))

        # SVC
        # clf = SVC(kernel="linear") # estimator
        # ExtraTreesClassifier
        clf = ExtraTreesClassifier(n_estimators=200, random_state=2222, class_weight='balanced')

        # # Ensemble
        # from sklearn.ensemble import VotingClassifier
        # from sklearn.linear_model import LogisticRegression
        # from sklearn.ensemble import GradientBoostingClassifier
        # from lightgbm import LGBMClassifier
        # from xgboost import XGBClassifier

        # clf1 = SVC(class_weight='balanced', kernel='rbf')
        # clf2 = LogisticRegression(multi_class='multinomial', penalty='l2', solver='saga')
        # clf3 = XGBClassifier(booster='gbtree', tree_method='auto', objective='reg:logistic')
        # clf4 = LGBMClassifier(boosting_type='goss')
        
        # eclf = VotingClassifier(estimators=[('svm', clf1), ('lr', clf2), ('xgb', clf3), ('lgbm', clf4)], voting='hard')

        for cv_num in cv_nums:
            eeg_selected_features = RFECV4AD(X=X_eeg, y=y_eeg, data_type='eeg', estimator=clf, step=1,
                                            loo_lda=False, cv_num=cv_num, min_features=10)
            fnirs_selected_features = RFECV4AD(X=X_fnirs, y=y_fnirs, data_type='fnirs', estimator=clf, step=1,
                                            loo_lda=False, cv_num=cv_num, min_features=10)

            ret[cv_num].append(len(eeg_selected_features))
            ret[cv_num].append(len(fnirs_selected_features))

        print(ret)

    else: # eeg + fnirs hybrid

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
        
        print('>> EEG:', len(eeg_features))
        print('>> FNIRS:', len(fnirs_features))
        hybrid_features = eeg_features + fnirs_features
        X_hybrid = X_total[hybrid_features]


        ####################################################################################################

        # RFECV
        cv_nums = [5, 10]
        ret = {k:[] for k in cv_nums}


        # # LOOCV + LDA
        # # loo = LeaveOneOut()
        # lda = LinearDiscriminantAnalysis()
        # # print(cross_val_score(estimator=lda, X=X_hybrid, y=y, cv=loo))

        # # RFECV4AD = RFECV for AD
        # hybrid_selected_features = RFECV4AD(X=X_hybrid, y=y, data_type='hybrid', estimator=lda, step=1,
        #                                 loo_lda=True, cv_num=0, min_features=30)

        # print(len(hybrid_selected_features))

        # SVC
        # clf = SVC(kernel="linear") # estimator
        # ExtraTreesClassifier
        clf = ExtraTreesClassifier(n_estimators=1000, random_state=2222, class_weight='balanced')

        for cv_num in cv_nums:
            hybrid_selected_features = RFECV4AD(X=X_hybrid, y=y, data_type='hybrid', estimator=clf, step=1,
                                            loo_lda=False, cv_num=cv_num, min_features=30)

            ret[cv_num].append(len(hybrid_selected_features))

        print(ret)

        

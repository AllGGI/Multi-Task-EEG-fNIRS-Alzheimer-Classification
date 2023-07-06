import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time
import datetime
from tqdm import tqdm

# RFECV (Recursive Feature Elimination with Cross-Validation to select features)
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV 

# Metrics
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# ML models
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC
# from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.ensemble import RandomForestClassifier

# MLP
from torch import optim, nn
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

# My files
from P4_v1_make_index import make_index
from P4_v1_get_intersection_of_optimal_features import get_intersection_of_folds


def RFECV4AD(X, y, data_type, estimator, step, cv_num, min_features, save_pth):

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=2222) # StratifiedKFold(cv_num)
    cv_name = 'SKF'

    rfecv = RFECV(
        estimator=estimator,
        step=step, # step = the number of features removed at each iteration
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

    # Select desired act cols
    only_this_act_cols = []
    for col in total_X.columns:
        act_part = int(col.split('_')[1][-1]) # 0-5
        
        if act_part in act_list: only_this_act_cols.append(col)

    # Split EEG / fNIRS col
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


class MLP(nn.Module):
    def __init__(self, num_class=3):
        super().__init__()

        self.fc1 = nn.Linear(43, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_class)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=1) # log_


class SimpleDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # x = torch.FloatTensor()
        # y = torch.LongTensor()

        return torch.FloatTensor(self.X[idx,:]), self.y[idx]



def rfecv_feature_selection(cv_num, act_num, pth, csv_root, save_root, this_seed):

    if not os.path.exists(save_root):
        os.makedirs(save_root)
    if not os.path.exists(csv_root):
        os.makedirs(csv_root)

    ####################################################################
    # 1) Make Train/Test split files before feature selection
    X_total, y = make_index(act_num, pth, csv_root, save_root)
    ####################################################################

    
    for fold_num in range(5):
        print('>> Fold:', fold_num)
        start_time = time()

        save_path = save_root + pth + '_' + '-'.join([str(i) for i in act_num]) + '/Fold_' + str(fold_num) + '/'
        print('>> save path:', save_path)

        ####################################################################
        # 3) Load Train/Test split files
        train_index = np.load(save_path + 'train_index.npy') # Fold 0 of StratifiedKfold
        test_index = np.load(save_path + 'test_index.npy')

        X_train, X_test = X_total.loc[train_index,:], X_total.loc[test_index,:]
        y_train, y_test = y.loc[train_index], y.loc[test_index]

        # print(y_test.to_list().count(0), y_test.to_list().count(1), y_test.to_list().count(2))
        # print(y_test.to_list())

        ####################################################################
        # 4) Get data of desired acts
        data_dict = multiple_acts(X_train, y_train, act_list=act_num, norm=False)
        
        
        ####################################################################
        # 5) RFECV (ExtraTreesClassifier) for EEG / fNIRS each
        ret = {cv_num:[]}
        clf = ExtraTreesClassifier(n_estimators=100, random_state=this_seed, class_weight='balanced', n_jobs=-1)
        
        eeg_selected_features, eeg_max_test_mean_score = RFECV4AD(X=data_dict['X_eeg'], y=data_dict['y_eeg'], data_type='eeg',
                                        estimator=clf, step=1, cv_num=cv_num, min_features=10, save_pth=save_path)
        fnirs_selected_features, fnirs_max_test_mean_score = RFECV4AD(X=data_dict['X_fnirs'], y=data_dict['y_fnirs'], data_type='fnirs',
                                        estimator=clf, step=1, cv_num=cv_num, min_features=10, save_pth=save_path)

        ret[cv_num].append(len(eeg_selected_features))
        ret[cv_num].append(eeg_max_test_mean_score)
        ret[cv_num].append(len(fnirs_selected_features))
        ret[cv_num].append(fnirs_max_test_mean_score)

        print(ret)
        

        ####################################################################
        # 6) RFECV (ExtraTreesClassifier) for EEG+fNIRS
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
        
        print('>> EEG:', len(eeg_features))
        print('>> FNIRS:', len(fnirs_features))
        hybrid_features = eeg_features + fnirs_features
        X_hybrid = X_total[hybrid_features]

        # NO RFECV for hybrid -> Use EEG+fNIRS
        # ret = {cv_num:[]}
        # clf = ExtraTreesClassifier(n_estimators=100, random_state=this_seed, class_weight='balanced', n_jobs=-1)

        # hybrid_selected_features, max_mean_test_score = RFECV4AD(X=X_hybrid, y=y, data_type='hybrid',
        #                                 estimator=clf, step=1, cv_num=cv_num, min_features=30, save_pth=save_path)

        # ret[cv_num].append(len(hybrid_selected_features))
        # ret[cv_num].append(max_mean_test_score)

        # print(ret)

        ####################################################################
        end_time = time()
        total_sec = end_time - start_time
        total_time = str(datetime.timedelta(seconds=total_sec)).split(".")
        print(">> Total time:", total_time[0], "\n")
        print('*'*100)



def classification(cv_num, act_num, pth, csv_root, save_root, this_seed, clf_type):

    if not os.path.exists(save_root):
        os.makedirs(save_root)
    if not os.path.exists(csv_root):
        os.makedirs(csv_root)

    ####################################################################
    # 1) Make Train/Test split files before feature selection
    X_total, y = make_index(act_num, pth, csv_root, save_root)
    ####################################################################

    ####################################################################
    # Test hybrid features for ALL Fold
    # act_types = {0:'R', 1:'C1', 2:'C2', 3:'N1', 4:'N2', 5:'V'}
    hybrid_all_fold = get_intersection_of_folds(act_num, pth, cv_num, save_root)


    f1_list, auc_list, acc_list = [], [], []
    total_confusion = np.zeros((3, 3))
    for fold_num in range(5):
        save_path = save_root + pth + '_' + '-'.join([str(i) for i in act_num]) + '/Fold_' + str(fold_num) + '/'
        # print('>> save path:', save_path)

        ####################################################################
        # 3) Load Train/Test split files
        train_index = np.load(save_path + 'train_index.npy') # Fold 0 of StratifiedKfold
        test_index = np.load(save_path + 'test_index.npy')

        X_train, X_test = X_total.loc[train_index,:], X_total.loc[test_index,:]
        y_train, y_test = y.loc[train_index], y.loc[test_index]

        # print(y_test.to_list().count(0), y_test.to_list().count(1), y_test.to_list().count(2))
        # print(y_test.to_list())

        ####################################################################
        # 4) Get selected optimal features
        X_train = X_train[hybrid_all_fold]
        X_test = X_test[hybrid_all_fold]

        # 5) Define classifier
        if clf_type != 'MLP':
            if clf_type == 'Tree': clf = ExtraTreesClassifier(n_estimators=100, random_state=this_seed, class_weight='balanced', n_jobs=-1) # 1263
            elif clf_type == 'SVM': clf = SVC(C=5, kernel='rbf', class_weight='balanced', decision_function_shape='ovr', probability=True)
            elif clf_type == 'RF': clf = RandomForestClassifier(n_estimators=100, random_state=this_seed, class_weight='balanced', n_jobs=-1)

            start_time = time()
            # 6) Train / Test
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            y_proba = clf.predict_proba(X_test)
            end_time = time()
            total_sec = end_time - start_time
            # print(">> Total sec:", total_sec)
            # print('*'*100)
            
            
            acc_list.append(np.round(accuracy_score(y_test, y_pred), 4))
            f1_list.append(np.round(f1_score(y_test, y_pred, average='macro'), 4))
            auc_list.append(np.round(roc_auc_score(y_test, y_proba, average='macro', multi_class='ovr'), 4))

            fold_confusion = confusion_matrix(y_test, y_pred)
            total_confusion += fold_confusion

        else: # MLP

            X_train, X_test = X_train.to_numpy(), X_test.to_numpy()
            y_train, y_test = y_train.to_numpy(), y_test.to_numpy()

            train_dataset = SimpleDataset(X_train, y_train)
            test_dataset = SimpleDataset(X_test, y_test)

            train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=5, shuffle=False)

            model = MLP(num_class=3)
            optimizer = optim.Adam(model.parameters(), lr=0.0005)
            criterion = nn.CrossEntropyLoss()


            patience = 10  # Early stopping patience; how long to wait after last time validation loss improved.
            best_val_loss = float('inf')
            best_f1_score = float('inf')
            best_auc_score = float('inf')
            best_acc = float('inf')
            best_pred, best_gt = [], []
            for epoch in range(1000):
                model.train()

                train_loss = 0
                train_total_preds, train_total_gt, train_total_proba = [], [], []
                for train_data, train_label in train_loader:

                    optimizer.zero_grad()
                    
                    output = model(train_data)
                    loss = criterion(output, train_label)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()

                    this_preds = torch.max(output, 1)[1]
                    train_total_preds += list(this_preds)
                    train_total_gt += list(train_label)
                    train_total_proba.append(output.detach().numpy())

                if epoch % 20 == 0:
                    with torch.no_grad():
                        model.eval()

                        test_loss = 0
                        test_total_preds, test_total_gt, test_total_proba = [], [], []
                        for test_data, test_label in test_loader:

                            test_output = model(test_data)
                            loss = criterion(test_output, test_label)
                            test_loss += loss.item()

                            this_preds = torch.max(test_output, 1)[1]
                            test_total_preds += list(this_preds)
                            test_total_gt += list(test_label)
                            test_total_proba.append(test_output.detach().numpy())

                    # print(np.asarray(test_total_gt))
                    # print(np.asarray(test_total_preds))

                    train_acc = accuracy_score(np.asarray(train_total_gt), np.asarray(train_total_preds))
                    train_f1 = f1_score(np.asarray(train_total_gt), np.asarray(train_total_preds), average='macro')
                    train_auc = roc_auc_score(np.asarray(train_total_gt), np.concatenate(train_total_proba), average='macro', multi_class='ovr')

                    test_acc = accuracy_score(np.asarray(test_total_gt), np.asarray(test_total_preds))
                    test_f1 = f1_score(np.asarray(test_total_gt), np.asarray(test_total_preds), average='macro')
                    test_auc = roc_auc_score(np.asarray(test_total_gt), np.concatenate(test_total_proba), average='macro', multi_class='ovr')

                    train_loss = train_loss / len(train_loader)
                    test_loss = test_loss / len(test_loader)

                    print('Epoch:', epoch,
                        '  Train loss: %.6f' % train_loss,
                        '  Test loss: %.6f' % test_loss,
                        '  Train F1 accuracy %.6f' % train_f1,
                        '  Train AUC accuracy %.6f' % train_auc,
                        '  Test F1 accuracy %.6f' % test_f1,
                        '  Test AUC accuracy %.6f' % test_auc
                    )

                    if test_loss < best_val_loss:
                        best_val_loss = test_loss
                        best_f1_score = test_f1
                        best_auc_score = test_auc
                        best_acc = test_acc
                        best_pred = test_total_preds
                        best_gt = test_total_gt
                        # torch.save(model.state_dict(), 'best_model.pt')
                        counter = 0
                    else:
                        counter += 1
                        if counter >= patience:
                            print('Early stopping')
                            # model.load_state_dict(torch.load('best_model.pt'))  # Load the best model
                            break

            
            print(
                'Best Acc %.6f' % best_acc,
                '  Best F1 accuracy %.6f' % best_f1_score,
                '  Best AUC accuracy %.6f' % best_auc_score,
            )
            f1_list.append(np.round(best_f1_score, 4))
            auc_list.append(np.round(best_auc_score, 4))
            acc_list.append(np.round(best_acc, 4))
            fold_confusion = confusion_matrix(best_gt, best_pred)
            total_confusion += fold_confusion

            print('='*100)


    
    print('\n>> Acc:', acc_list)
    print('>> F1:', f1_list)
    print('>> AUC:', auc_list)
    print('>> Mean Acc:', np.round(np.mean(acc_list), 4))
    print('>> Mean F1:', np.round(np.mean(f1_list), 4))
    print('>> Mean AUC:', np.round(np.mean(auc_list), 4))
    print('>> Confusion Matrix:')
    print(total_confusion)
    disp = ConfusionMatrixDisplay(total_confusion) # AD: 35 | CN: 63 | MCI: 46
    disp.plot()
    plt.show()




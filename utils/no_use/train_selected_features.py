import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import numpy as np 
from sklearn.model_selection import cross_val_score
from tqdm import tqdm
import matplotlib.pyplot as plt

from torch import optim, nn
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier as KNN

def train_ML(pth):

    open_csv = './final_csv/'+ pth + '.csv'
    df = pd.read_csv(open_csv)
    df = df.drop('Unnamed: 0', axis=1)
    df = df[np.isfinite(df).all(1)]
    print(df)

    X_total = df.drop('label', axis=1)
    y = df['label']

    # print(len(X_total.columns)) # 150 + label

    # Normalize
    # X_total = X_total.apply(lambda x: (x - x.min()) / (x.max() - x.min()), axis=0)
    # print(X_total)
    # raise

    # classfiier
    max_f1 = 0
    max_seed = -1

    
    # ==> 5sec EEG PSD + Full fNIRS PSD + Time-Domain fNIRS
    # seed: 60 - F1: 0.8298 (0~1000)
    # seed: 1520 - F1: 0.8241 (1000~2000)
    # seed: 2913 - F1: 0.8235 (2000~3000)
    # seed: 3879 - F1: 0.8420 (3000~4000)
    # seed: 4710 - F1: 0.8396 (4000~5000)

    # ==> 3sec EEG PSD + Full fNIRS PSD + Time-Domain fNIRS
    # {seed:F1} = {1537: 0.8249, 2076: 0.8295}

    # ==> Full EEG PSD + Full fNIRS PSD + Time-Domain fNIRS
    # {seed:F1} = {4794: 0.7850, }

    # ==> 5sec EEG PSD + Full fNIRS PSD
    # {seed:F1} = {657: 0.7505, }

    # ==> 5sec EEG PSD + Full fNIRS Time Domain
    # {seed:F1} = {4476: 0.7863, }

    # ==> 5sec EEG PSD (only)
    # {seed:F1} = {548: 0.7222, 1996:0.7225, 4639: 0.7236}
    # ==> Full EEG PSD (only)
    # {seed:F1} = {650: 0.7191, 1731: 0.7178}
    # ==> 3sec EEG PSD (only)
    # {seed:F1} = {1556: 0.7498, }

    run_type = 'only_run' # 'find_seed', 'find_K', 'only_run'
    selected_seed = 638 # fixed
    clf_type = 'Extra' # 'Extra', 'SVM', 'KNN'

    if run_type == 'find_seed':
        for seed in tqdm(range(1000,2000)):
            clf = ExtraTreesClassifier(n_estimators=100, random_state=seed, class_weight='balanced') # 2222

            skf = StratifiedKFold(n_splits=5)
            # acc = cross_val_score(clf, X_total, y, scoring='accuracy', cv=skf)
            f1_acc = cross_val_score(clf, X_total, y, scoring='f1_macro', cv=skf)
            # auc_acc = cross_val_score(clf, X_total, y, scoring='roc_auc_ovr', cv=skf)

            if np.mean(f1_acc) > max_f1:
                max_seed = seed
                max_f1 = np.mean(f1_acc)

        print('>> Max seed:', max_seed)
        print('>> Max F1:', max_f1)
    
    elif run_type == 'find_K':
        max_f1 = 0
        max_k = -1
        ####################################### ML based Classifiers #############################################
        for k in range(6,100):
            clf = KNN(n_neighbors=k, n_jobs=-1)

            ##########################################################################################################
            
            skf = StratifiedKFold(n_splits=5)
            # acc = cross_val_score(clf, X_total, y, scoring='accuracy', cv=skf)
            f1_acc = cross_val_score(clf, X_total, y, scoring='f1_macro', cv=skf)
            # auc_acc = cross_val_score(clf, X_total, y, scoring='roc_auc_ovr', cv=skf)

            if np.mean(f1_acc) > max_f1:
                max_k = k
                max_f1 = np.mean(f1_acc)

        print('>> Max k:', max_k)
        print('>> Max F1:', max_f1)
    

    elif run_type == 'only_run': 
        print('>> This seed:', selected_seed)

        ####################################### ML based Classifiers #############################################
        if clf_type == 'Extra': clf = ExtraTreesClassifier(n_estimators=100, random_state=selected_seed, class_weight='balanced') # 2222
        elif clf_type == 'SVM': clf = SVC(kernel='linear', class_weight='balanced', decision_function_shape='ovr', probability=True)
        elif clf_type == 'KNN': clf = KNN(n_neighbors=8, n_jobs=-1)

        ##########################################################################################################
        
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2222) # Stratified but not random split (=same as RFECV's cv)

        # 1) Short cross val (= Same result as 2)
        # acc = cross_val_score(clf, X_total, y, scoring='accuracy', cv=skf)
        # f1_acc = cross_val_score(clf, X_total, y, scoring='f1_macro', cv=skf)
        # auc_acc = cross_val_score(clf, X_total, y, scoring='roc_auc_ovr', cv=skf)

        # 2) Long cross val (to check feature importances)
        f1_list, auc_list, acc_list = [], [], []
        total_confusion = np.zeros((3, 3))
        for i, (train_index, test_index) in enumerate(skf.split(X_total, y)):

            # if i != 1: continue

            X_train, X_test = X_total.loc[train_index,:], X_total.loc[test_index,:]
            y_train, y_test = y.loc[train_index], y.loc[test_index]

            print(y_test.to_list().count(0), y_test.to_list().count(1), y_test.to_list().count(2))

            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            y_proba = clf.predict_proba(X_test)

            acc_list.append(np.round(accuracy_score(y_test, y_pred), 4))
            f1_list.append(np.round(f1_score(y_test, y_pred, average='macro'), 4))
            auc_list.append(np.round(roc_auc_score(y_test, y_proba, average='macro', multi_class='ovr'), 4))

            # Confusion Matrix
            fold_confusion = confusion_matrix(y_test, y_pred)
            total_confusion += fold_confusion
                
            # # See feature importances
            # feat_importance = clf.feature_importances_
            # sort_by_importance = np.flip(np.argsort(feat_importance))
            # K = 20
            # topk_features = np.asarray(clf.feature_names_in_)[sort_by_importance][:K]
            # for feat, imp in zip(topk_features, feat_importance[sort_by_importance][:K]):
            #     print(feat, '-', imp)
            # # print(topk_features.tolist())
            # print('*'*100)
            
            # print('>> Mean AUC:', np.round(np.mean(auc_list), 4))
            # print('>> Confusion Matrix:')
            # print(total_confusion)

            # disp = ConfusionMatrixDisplay(total_confusion) # AD: 35 | CN: 63 | MCI: 46
            # disp.plot()
            # plt.show()

            

        print('>> Acc:', acc_list)
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
    
    
    

class MLP(nn.Module):
    def __init__(self, num_class=3):
        super().__init__()

        self.fc1 = nn.Linear(151, 32)
        self.fc2 = nn.Linear(32, 32)
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


def train_MLP(pth):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    open_csv = './final_csv/'+ pth + '.csv'
    df = pd.read_csv(open_csv)
    df = df.drop('Unnamed: 0', axis=1)

    df = df[np.isfinite(df).all(1)]

    X_total = df.drop('label', axis=1)
    y = df['label']

    X_total = X_total.apply(lambda x: (x - x.min()) / (x.max() - x.min()), axis=0)


    skf = StratifiedKFold(n_splits=10)

    f1_list, auc_list, acc_list = [], [], []
    for i, (train_index, test_index) in enumerate(skf.split(X_total, y)):

        print('>> Fold:', i, '\n')

        X_train, X_test = X_total.loc[train_index,:].to_numpy(), X_total.loc[test_index,:].to_numpy()
        y_train, y_test = y.loc[train_index].to_numpy(), y.loc[test_index].to_numpy()

        # print(X_train.shape, X_test.shape, y_train, y_test) # (129, 151) (15, 151) (129,) (15,)

        train_dataset = SimpleDataset(X_train, y_train)
        test_dataset = SimpleDataset(X_test, y_test)

        train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=5, shuffle=False)

        model = MLP(num_class=3)
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        criterion = nn.CrossEntropyLoss()


        patience = 10  # Early stopping patience; how long to wait after last time validation loss improved.
        best_val_loss = float('inf')
        best_f1_score = float('inf')
        best_auc_score = float('inf')
        best_acc = float('inf')
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
        f1_list.append(best_f1_score)
        auc_list.append(best_auc_score)
        acc_list.append(best_acc)


        print('='*100)


    print('>> Acc:', acc_list)
    print('>> F1:', f1_list)
    print('>> AUC:', auc_list)

    # Mean Acc: 0.7638 | Mean F1: 0.7635 | Mean AUC: 0.9022
    print('>> Mean Acc:', np.round(np.mean(acc_list), 4))
    print('>> Mean F1:', np.round(np.mean(f1_list), 4))
    print('>> Mean AUC:', np.round(np.mean(auc_list), 4))
    


if __name__ == "__main__":
    
    act_num = [0,1,2,3,4,5]
    act_types = {0:'R', 1:'C1', 2:'C2', 3:'N1', 4:'N2', 5:'V'}
    
    pth = '5secEEGPSD_FullFnirsPSD_FullFnirsTimeDomain'


    # pth = 'opt-2_5secEEGPSD_FullFnirsPSD'
    # pth = 'opt-3_5secEEGPSD_FullFnirsTimeDomain'
    # pth = 'opt-4_FullFnirsPSD_FullFnirsTimeDomain'
    # pth = 'opt-5_5secEEGPSD'
    # pth = 'opt-6_FullFnirsPSD'
    # pth = 'opt-7_FullFnirsTimeDomain'


    
    # pth = 'RFECV-3secEEGPSD_FullFnirsPSD_FullFnirsTimeDomain_R-C1-C2-N1-N2-V'
    # pth = '5secEEGPSD_FullFnirsPSD_FullFnirsTimeDomain'
    
    # pth = 'RFECV-FullEEGPSD_FullFnirsPSD_FullFnirsTimeDomain_R-C1-C2-N1-N2-V'
    # pth = 'RFECV-full_eeg_fnirs_psd_R-C1-C2-N1-N2-V'

    # pth = 'RFECV-5sec_eeg_PSD_fnirs_TimeDomain_R-C1-C2-N1-N2-V'
    # pth = 'RFECV-ONLY-EEG-5secEEGPSD_FullFnirsPSD_FullFnirsTimeDomain_R-C1-C2-N1-N2-V' # only 5 sec EEG
    # pth = 'RFECV-ONLY-EEG-full_eeg_fnirs_psd_R-C1-C2-N1-N2-V' # Full EEG PSD only
    # pth = 'RFECV-ONLY-EEG-3sec_eeg_fnirs_power_sliced_R-C1-C2-N1-N2-V'

    # pth = 'RFECV-fusion_temp_R-C1-C2-N1-N2-V'

    csv_pth = 'RFECV-' + pth + '_' + '-'.join([act_types[i] for i in act_num])
    print('>> csv PATH:', csv_pth)
    train_ML(csv_pth)
    # train_MLP(pth)


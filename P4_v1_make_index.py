import pandas as pd
from sklearn.model_selection import StratifiedKFold
import numpy as np
import os

def make_index(act_num, pth, csv_root, save_root):

    open_csv = csv_root + pth + '.csv'

    df = pd.read_csv(open_csv)
    df = df.drop('Unnamed: 0', axis=1)
    df = df[np.isfinite(df).all(1)]
    # print('>> Dataframe:\n')
    # print(df)

    X_total = df.drop('label', axis=1)
    y = df['label']

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2222) # Stratified but not random split (=same as RFECV's cv)


    for fold_idx, (train_index, test_index) in enumerate(skf.split(X_total, y)):

        # X_train, X_test = X_total.loc[train_index,:], X_total.loc[test_index,:]
        # y_train, y_test = y.loc[train_index], y.loc[test_index]

        save_path = save_root + pth + '_' + '-'.join([str(i) for i in act_num]) + '/Fold_' + str(fold_idx) + '/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # print(save_path)
        # print(y_test.to_list().count(0), y_test.to_list().count(1), y_test.to_list().count(2))
        # print(test_index)

        np.save(save_path + 'train_index.npy', train_index)
        np.save(save_path + 'test_index.npy', test_index)

        # print('*'*100)

        
    return X_total, y



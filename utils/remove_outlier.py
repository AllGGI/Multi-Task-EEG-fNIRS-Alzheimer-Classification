import numpy as np
import pandas as pd
from sklearn import svm, neighbors, ensemble

pth = 'csv/new_eeg_fnirs_power_sliced_meansumstd.csv'

df = pd.read_csv(pth)
df = df.drop('Unnamed: 0', axis=1)
df = df[np.isfinite(df).all(1)]

X_total = df.drop('label', axis=1)
y = df['label']

# print(X_total)

list_y = list(y)
# print(list(y).count(0), list(y).count(2))
# print(list_y[25], list_y[26], list_y[-47], list_y[-46]) # 26 ~ -46(exclude)

norm_X = X_total.iloc[26:-46]
norm_y = y.iloc[26:-46]

# print(norm_X)

ad_X = X_total.iloc[:26]
ad_y = y.iloc[:26]

mci_X = X_total.iloc[-46:]
mci_y = y.iloc[-46:]

print(list_y[26:-46])
print(list_y[:26])
print(list_y[-46:])

######################################################################
# Finding outlier method 1)
# clf = neighbors.LocalOutlierFactor(n_neighbors=40)
# print(clf.fit_predict(norm_X))

# Finding outlier method 2)
ad_clf = ensemble.IsolationForest(n_estimators=1000)
# print(ad_clf.fit_predict(ad_X))
ad_idx = ad_clf.fit_predict(ad_X).tolist()

norm_clf = ensemble.IsolationForest(n_estimators=100)
print(norm_clf.fit_predict(norm_X))
norm_idx = norm_clf.fit_predict(norm_X).tolist()

mci_clf = ensemble.IsolationForest(n_estimators=1000)
# print(mci_clf.fit_predict(mci_X))
mci_idx = mci_clf.fit_predict(mci_X).tolist()

idx = ad_idx + norm_idx + mci_idx # outlier index
# print(idx)

outliers = []
for i, val in enumerate(idx):
    if val == -1: outliers.append(i)

print(outliers) # index of outliers

good = list(set(range(len(X_total)))-set(outliers)) # index of not outliers
print(good)


new_X = X_total.iloc[good]
new_y = y.iloc[good]
# print(new_X)
# print(new_y)
new_X.insert(0, 'label', new_y)
# print(new_X)
new_X.to_csv('./csv/new_eeg_fnirs_power_sliced_meansumstd_outlier-removed.csv', sep=',') # outlier removed csv


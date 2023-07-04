import pandas as pd
import numpy as np


pth = 'RFECV-5secEEGPSD_FullFnirsPSD_FullFnirsTimeDomain_R-C1-C2-N1-N2-V'
open_csv = './final_csv/'+ pth + '.csv'
df = pd.read_csv(open_csv)
df = df.drop('Unnamed: 0', axis=1)
df = df[np.isfinite(df).all(1)]
print(df)

X_total = df.drop('label', axis=1)
y = df['label']

feature_categories = {'eeg':[], 'fnirs_freq':[], 'fnirs_time':[]}
feature_counts = {'eeg':0, 'fnirs_freq':0, 'fnirs_time':0}
for feat in X_total.columns:
    if 'eeg' in feat:
        feature_counts['eeg'] += 1
        feature_categories['eeg'].append(feat)
    elif ('fnirs' in feat) and ('time' in feat):
        feature_counts['fnirs_time'] += 1
        feature_categories['fnirs_time'].append(feat)
    elif ('fnirs' in feat) and ('band' in feat):
        feature_counts['fnirs_freq'] += 1
        feature_categories['fnirs_freq'].append(feat)
    else: raise NotImplementedError

assert (feature_counts['eeg'] + feature_counts['fnirs_freq'] + feature_counts['fnirs_time']) == 151

print(feature_counts)

eeg_bands = ['Alpha', 'Beta', 'Theta', 'Delta', 'Gamma']
eeg_feat_dict = {i:{b:0 for b in eeg_bands} for i in range(4)}
# print(eeg_feat_dict)
for eeg_feat in feature_categories['eeg']:
    temp = eeg_feat.split('_')
    act = int(temp[1].split('-')[-1])
    if act == 1 or act == 2: act = 1
    elif act == 3 or act == 4: act = 2
    elif act == 5: act = 3
    band = temp[3].split('-')[-1]
    if 'Gamma' in band: band = 'Gamma'
    # print(act, band) # 0, Delta
    eeg_feat_dict[act][band] += 1

print(eeg_feat_dict)


fnirs_feat_dict = {i:0 for i in range(4)}
for fnirs_feat in feature_categories['fnirs_freq']:
    temp = fnirs_feat.split('_') # fnirs_act-1_HbO_feat-Px_band-vlfo_ch-4
    act = int(temp[1].split('-')[-1])
    hb_type = temp[2]

    if act == 1 or act == 2: act = 1
    elif act == 3 or act == 4: act = 2
    elif act == 5: act = 3
    fnirs_feat_dict[act] += 1

print(fnirs_feat_dict)


fnirs_feat_dict = {i:0 for i in range(4)}
for fnirs_feat in feature_categories['fnirs_time']:
    temp = fnirs_feat.split('_') # fnirs_act-1_HbO_feat-Px_band-vlfo_ch-4
    act = int(temp[1].split('-')[-1])
    hb_type = temp[2]

    if act == 1 or act == 2: act = 1
    elif act == 3 or act == 4: act = 2
    elif act == 5: act = 3
    fnirs_feat_dict[act] += 1

print(fnirs_feat_dict)



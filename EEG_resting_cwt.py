import os
import scipy
from scipy import io
import numpy as np
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import pywt

action = 'Resting'

save_root = './Experiments/' + action + '/EEG/new_scale/input_normalized/'
eeg_path = "D:\치매감지\EEG_fNIRs_dataset\Sorted_Dataset\Sorted_alzh_dataset/".replace('\\', '/')
eeg_levels = ['AD', 'NORMAL', 'NP', 'MCI']
eeg_data_list = {'AD':[], 'NORMAL':[], 'NP':[], 'MCI':[]} # ad, cn, ns, pre order

for i, level in enumerate(eeg_levels):
    patient_list = glob(os.path.join(eeg_path, level, '*'))
    eeg_data_list[level] += [sorted(glob(os.path.join(x, 'EEG/*RO.mat')))[0] for x in patient_list]


window = 5 # 5 sec
sampling_rate = 500
one_sample_size = sampling_rate * window # 5 sec per sample
eeg_bands = {'Delta': (1, 4),
                'Theta': (4, 8),
                'Alpha': (8, 13),
                'Beta': (13, 30),
                'Gamma': (30, 70)} # (70 -> 40)

# Get scales
freq = np.array([1, 4, 8, 13, 30, 70]) / sampling_rate
scale = pywt.frequency2scale('mexh', freq)
eeg_scales = [round(s) for s in scale] # [125, 31, 16, 10, 4, 2]
eeg_band2scale = {'Delta': (eeg_scales[1], eeg_scales[0]),
                'Theta': (eeg_scales[2], eeg_scales[1]),
                'Alpha': (eeg_scales[3], eeg_scales[2]),
                'Beta': (eeg_scales[4], eeg_scales[3]),
                'Gamma': (eeg_scales[5], eeg_scales[4])} # 70


def plot_scalogram(coef, n_samples=1000, n_scale=70):
    plt.figure(figsize=(15,10))
    plt.imshow(abs(coef), extent=[0, n_samples, n_scale, 1], interpolation='bilinear', cmap='bone', aspect='auto', vmax=abs(coef).max(), vmin=-abs(coef).max())
    plt.gca().invert_yaxis()
    # plt.yticks(np.arange(1,n_scale,1))
    # plt.xticks(np.arange(0,samples,10))
    plt.show()


def Method1():
    '''
        Method 1)
        
        에폭마다 TFR 만듬 -> 에폭들 평균내서 ATFR 만듬 ->
        각 19채널마다 (coef, time) 2d array에서 feature (4 band * 3)개 뽑음
    
        => 환자 * (32ch * 15feat)

    '''
    scales = np.arange(1, 150, 1)
    epochs = 12 # int(data.shape[0] / one_sample_size) 
    for level in eeg_levels:

        if level == 'NP': continue

        patients = []
        for RO_file in eeg_data_list[level]:
            data = io.loadmat(RO_file)['data']
            # print(data.shape) # most: (31750, 32)

            channel_acwts = []
            for ch in range(32):
                total_epochs_cwt = []
                for epoch in range(epochs):
                    epoch_data = data[one_sample_size*epoch:one_sample_size*(epoch+1), ch]

                    # normalization
                    norm = np.linalg.norm(epoch_data)
                    epoch_data = epoch_data/norm

                    # CWT
                    coef, freqs = pywt.cwt(epoch_data, scales, 'mexh')
                    # print(coef.shape) # (149, 2500)

                    # plot_scalogram(coef, n_samples=samples, n_scale=eeg_scale)
                    total_epochs_cwt.append(coef)
                    # print(coef.shape) # (eeg_scale, one_sample_size) = (70, 2500)
                total_epochs_cwt = np.stack(total_epochs_cwt, axis=0)
                acwt = np.mean(total_epochs_cwt, axis=0)
                channel_acwts.append(acwt)
            
            this_file_acwt = np.stack(channel_acwts, axis=0) # (32, 70, 2500)

            # 1~71
            features = [] # 15 * 32 channel
            for ch in range(32):
                for band in eeg_band2scale.keys():
                    band_data = this_file_acwt[ch, eeg_band2scale[band][0]:eeg_band2scale[band][1], :] # 0~3 / 3~7 / 7~12 / 12~29 / 29~69
                    
                    mu = np.mean(band_data, axis=None) # (2,4)
                    std = np.std(band_data, axis=None)
                    skew = scipy.stats.skew(band_data, axis=None)
                    kurtosis = scipy.stats.kurtosis(band_data, axis=None)

                    # print(mu, std, skew)
                    features += [mu, std, skew, kurtosis]

                # print(len(features)) # 15

            # print(len(features)) # 15 * 32 = 480
            patients.append(features)
              
        result_feat = np.asarray(patients)
        print(level, '->', result_feat.shape) # (patient num, 480)

        save_path = save_root + 'Method1/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        np.save(save_path + level + '_input.npy', result_feat)


def Method2():
    
        # Method 2)
        
        # 각 에폭마다(5초마다) (현재 E번째 에폭)

        # 32개 채널 채널마다 TFR 생성 -> 32개 TFR -> 평균내서 1개 TFR
        # E번째 ATFR에서 mean, std, skewness 뽑음 (각 band에 대해) -> 15개 feature

        # 이거로 MLP

        # => 저장을 에폭 단위로 해야함: 환자 * 12 epoch * 15 features
    
    scales = np.arange(1, 150, 1)
    epochs = 12 # int(data.shape[0] / one_sample_size)
    for level in eeg_levels:
        patients = []
        for RO_file in eeg_data_list[level]:
            data = io.loadmat(RO_file)['data']
            # print(data.shape) # most: (31750, 32)

            features = []
            for epoch in range(epochs):
                channel_acwts = []
                for ch in range(32):
                    epoch_data = data[one_sample_size*epoch:one_sample_size*(epoch+1), ch] # (2500,)

                    # normalization
                    norm = np.linalg.norm(epoch_data)
                    epoch_data = epoch_data/norm

                    # CWT
                    coef, freqs = pywt.cwt(epoch_data, scales, 'mexh')

                    # plot_scalogram(coef, n_samples=samples, n_scale=eeg_scale)
                    channel_acwts.append(coef)
                    # print(coef.shape) # (eeg_scale, one_sample_size) = (70, 2500)
                channel_acwts = np.stack(channel_acwts, axis=0) # (32, 70, 2500)

                # Average of all channel's TFR -> 1 TFR
                acwt = np.mean(channel_acwts, axis=0) # (70, 2500)

                # Get mean/std/skewness/(kurtosis) features for each band
                band_features = [] # 15
                for band in eeg_bands.keys():
                    band_data = acwt[eeg_band2scale[band][0]:eeg_band2scale[band][1], :] # 0~3 / 3~7 / 7~12 / 12~29 / 29~69
                    # print(band_data.shape) # (3, 2500)

                    mu = np.mean(band_data, axis=None) # (2,4)
                    std = np.std(band_data, axis=None)
                    skew = scipy.stats.skew(band_data, axis=None)
                    # kurtosis = scipy.stats.kurtosis(band_data, axis=None)


                    # print(mu, std, skew)
                    band_features += [mu, std, skew]
                    # band_features += [mu, std, skew, kurtosis]

                features.append(band_features)

            # features = np.asarray(features)
            # print(features.shape) # (12, 15)
            patients.append(features)
              
        result_feat = np.asarray(patients)
        print(level, '->', result_feat.shape) # (patient num, 12, 15)

        save_path = save_root + 'Method2/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        np.save(save_path + level + '_input.npy', result_feat)

'''

def Method3():

        # Method 3)

        # 환자별로 (32, 70, 2500)을 그대로 저장

    scales = np.arange(1, eeg_scale+1, 1)
    epochs = 12 # int(data.shape[0] / one_sample_size) 
    for level in eeg_levels:
        patients = []
        for RO_file in eeg_data_list[level]:
            data = io.loadmat(RO_file)['data']
            # print(data.shape) # most: (31750, 32)

            channel_acwts = []
            for ch in range(32):
                total_epochs_cwt = []
                for epoch in range(epochs):
                    epoch_data = data[one_sample_size*epoch:one_sample_size*(epoch+1), ch]

                    # normalization
                    norm = np.linalg.norm(epoch_data)
                    epoch_data = epoch_data/norm

                    # CWT
                    coef, freqs = pywt.cwt(epoch_data, scales, 'mexh')

                    # plot_scalogram(coef, n_samples=samples, n_scale=eeg_scale)
                    total_epochs_cwt.append(coef)
                    # print(coef.shape) # (eeg_scale, one_sample_size) = (70, 2500)
                total_epochs_cwt = np.stack(total_epochs_cwt, axis=0)
                acwt = np.mean(total_epochs_cwt, axis=0)
                channel_acwts.append(acwt)
            
            this_file_acwt = np.stack(channel_acwts, axis=0) # (32, 70, 2500)
            patients.append(this_file_acwt)
              
        result_feat = np.stack(patients, axis=0)
        print(level, '->', result_feat.shape) # (patient num, 32, 70, 2500)

        save_path = save_root + 'Method3/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        np.save(save_path + level + '_input.npy', result_feat)

'''

if __name__ == "__main__":

    Method1()
    # Method2()
    # Method3()
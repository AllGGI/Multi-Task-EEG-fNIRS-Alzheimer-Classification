import os
import scipy
from scipy import io
import numpy as np
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import pywt

action = 'Resting'

save_root = './Experiments/' + action + '/FNIRS/new_scale/input_normalized/'
fNIRs_path = "D:\치매감지\EEG_fNIRs_dataset\Sorted_Dataset\Sorted_alzh_dataset/".replace('\\', '/')
fNIRs_levels = ['AD', 'NORMAL', 'NP', 'MCI']
fNIRs_data_list = {'AD':[], 'NORMAL':[], 'NP':[], 'MCI':[]} # ad, cn, ns, pre order

for i, level in enumerate(fNIRs_levels):
    patient_list = glob(os.path.join(fNIRs_path, level, '*'))
    fNIRs_data_list[level] += [sorted(glob(os.path.join(x, 'fNIRs/p_01_resting.mat')))[0] for x in patient_list]

sampling_rate = 8
freq = np.array([0.01, 0.04, 0.15, 0.16, 0.6, 0.8, 1.5]) / sampling_rate
scale = pywt.frequency2scale('morl', freq)
fnirs_scales = [round(s) for s in scale] # [650, 162, 43, 41, 11, 8, 4]

fnirs_band = {
                'VLFO': (0.01, 0.04),
                'LFO': (0.04, 0.15),
                'Respiration': (0.16, 0.6),
                'Heartbeat': (0.8, 1.5)
            }

fnirs_band2scale = {
                'VLFO': (fnirs_scales[1], fnirs_scales[0]),
                'LFO': (fnirs_scales[2], fnirs_scales[1]),
                'Respiration': (fnirs_scales[4], fnirs_scales[3]),
                'Heartbeat': (fnirs_scales[6], fnirs_scales[5])
            }

window = 5
one_sample_size = sampling_rate * window # 5 sec per sample


# def plot_scalogram(coef, n_samples=1000, n_scale=70):
#     plt.figure(figsize=(15,10))
#     plt.imshow(abs(coef), extent=[0, n_samples, n_scale, 1], interpolation='bilinear', cmap='bone', aspect='auto', vmax=abs(coef).max(), vmin=-abs(coef).max())
#     plt.gca().invert_yaxis()
#     # plt.yticks(np.arange(1,n_scale,1))
#     # plt.xticks(np.arange(0,samples,10))
#     plt.show()


def Method1():
    '''
        Method 1)
        
        에폭마다 TFR 만듬 -> 에폭들 평균내서 ATFR 만듬 ->
        각 19채널마다 (coef, time) 2d array에서 feature (4 band * 3)개 뽑음
    
        => 환자 * (32ch * 15feat)

    '''
    scales = np.arange(1, 700, 1)
    epochs = 12 # 12 * 5sec = 60sec, int(Hb.shape[0] / one_sample_size) 
    for level in fNIRs_levels:
        if level == 'NP': continue

        patients_Hb = []
        patients_HbO = []
        for RO_file in fNIRs_data_list[level]:
            Hb = io.loadmat(RO_file)['Hb'] # (567, 6)
            HbO = io.loadmat(RO_file)['HbO'] # (567, 6)
            # THb = io.loadmat(RO_file)['THb'] # (567, 6)
            # print(Hb.shape, HbO.shape, THb.shape)

            Hb_channel_acwts = []
            HbO_channel_acwts = []
            for ch in range(6):
                Hb_total_epochs_cwt = []
                HbO_total_epochs_cwt = []
                for epoch in range(epochs):
                    Hb_epoch_data = Hb[one_sample_size*epoch:one_sample_size*(epoch+1), ch]
                    HbO_epoch_data = HbO[one_sample_size*epoch:one_sample_size*(epoch+1), ch]

                    # Hb
                    # normalization
                    norm = np.linalg.norm(Hb_epoch_data)
                    Hb_epoch_data = Hb_epoch_data/norm
                    # CWT
                    Hb_coef, Hb_freqs = pywt.cwt(Hb_epoch_data, scales, 'morl')

                    # HbO
                    # normalization
                    norm = np.linalg.norm(HbO_epoch_data)
                    HbO_epoch_data = HbO_epoch_data/norm
                    # CWT
                    HbO_coef, HbO_freqs = pywt.cwt(HbO_epoch_data, scales, 'morl')

                    # plot_scalogram(coef, n_samples=samples, n_scale=fNIRs_scale)
                    Hb_total_epochs_cwt.append(Hb_coef)
                    HbO_total_epochs_cwt.append(HbO_coef)
                    
                    # print(coef.shape) # (fNIRs_scale, one_sample_size) = (699, 40)
                Hb_total_epochs_cwt = np.stack(Hb_total_epochs_cwt, axis=0)
                Hb_acwt = np.mean(Hb_total_epochs_cwt, axis=0)
                HbO_total_epochs_cwt = np.stack(HbO_total_epochs_cwt, axis=0)
                HbO_acwt = np.mean(HbO_total_epochs_cwt, axis=0)

                Hb_channel_acwts.append(Hb_acwt)
                HbO_channel_acwts.append(HbO_acwt)
            
            this_file_acwt_Hb = np.stack(Hb_channel_acwts, axis=0) # (6, 699, 40)
            this_file_acwt_HbO = np.stack(HbO_channel_acwts, axis=0) # (6, 699, 40)
            
            # Hb
            features = [] # 15 * 6 channel
            for ch in range(6):
                for band in fnirs_band2scale.keys():
                    band_data = this_file_acwt_Hb[ch, fnirs_band2scale[band][0]:fnirs_band2scale[band][1], :]

                    mu = np.mean(band_data, axis=None) # (2,4)
                    std = np.std(band_data, axis=None)
                    skew = scipy.stats.skew(band_data, axis=None)
                    kurtosis = scipy.stats.kurtosis(band_data, axis=None)

                    # print(mu, std, skew)
                    features += [mu, std, skew, kurtosis]

                # print(len(features)) # 15

            # print(len(features)) # 15 * 32 = 480
            patients_Hb.append(features)
            

            # HbO
            features = [] # 15 * 6 channel
            for ch in range(6):
                for band in fnirs_band2scale.keys():
                    band_data = this_file_acwt_HbO[ch, fnirs_band2scale[band][0]:fnirs_band2scale[band][1], :]

                    mu = np.mean(band_data, axis=None) # (2,4)
                    std = np.std(band_data, axis=None)
                    skew = scipy.stats.skew(band_data, axis=None)
                    kurtosis = scipy.stats.kurtosis(band_data, axis=None)

                    # print(mu, std, skew)
                    features += [mu, std, skew, kurtosis]

                # print(len(features)) # 15

            # print(len(features)) # 15 * 32 = 480
            patients_HbO.append(features)

        result_feat_Hb = np.asarray(patients_Hb)
        print(level, '->', result_feat_Hb.shape) # (patient num, 120) : 4band * 5feat * 6ch
        result_feat_HbO = np.asarray(patients_HbO)
        print(level, '->', result_feat_HbO.shape) # (patient num, 480)

        save_path = save_root + 'Method1/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        np.save(save_path + level + '_input_Hb.npy', result_feat_Hb)
        np.save(save_path + level + '_input_HbO.npy', result_feat_HbO)
        


def Method2():
    '''
        Method 2)
        
        각 에폭마다(5초마다) (현재 E번째 에폭)

        32개 채널 채널마다 TFR 생성 -> 32개 TFR -> 평균내서 1개 TFR
        E번째 ATFR에서 mean, std, skewness 뽑음 (각 band에 대해) -> 15개 feature

        이거로 MLP

        => 저장을 에폭 단위로 해야함: 환자 * 12 epoch * 15 features
    '''
    scales = np.arange(1, 700, 1)
    epochs = 12 # int(data.shape[0] / one_sample_size)
    for level in fNIRs_levels:
        patients = []
        for RO_file in fNIRs_data_list[level]:
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
                    coef, freqs = pywt.cwt(epoch_data, scales, 'morl')

                    # plot_scalogram(coef, n_samples=samples, n_scale=fNIRs_scale)
                    channel_acwts.append(coef)
                    # print(coef.shape) # (fNIRs_scale, one_sample_size) = (70, 2500)
                channel_acwts = np.stack(channel_acwts, axis=0) # (32, 70, 2500)

                # Average of all channel's TFR -> 1 TFR
                acwt = np.mean(channel_acwts, axis=0) # (70, 2500)

                # Get mean/std/skewness/(kurtosis) features for each band
                band_features = [] # 15
                for band in fnirs_band2scale.keys():
                    band_data = acwt[fnirs_band2scale[band][0]:fnirs_band2scale[band][1], :] # 0~3 / 3~7 / 7~12 / 12~29 / 29~69
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


# def Method3():
#     '''
#         Method 3)

#         환자별로 (32, 70, 2500)을 그대로 저장

#     '''
#     scales = np.arange(1, fNIRs_scale+1, 1)
#     epochs = 12 # int(data.shape[0] / one_sample_size) 
#     for level in fNIRs_levels:
#         patients = []
#         for RO_file in fNIRs_data_list[level]:
#             data = io.loadmat(RO_file)['data']
#             # print(data.shape) # most: (31750, 32)

#             channel_acwts = []
#             for ch in range(32):
#                 total_epochs_cwt = []
#                 for epoch in range(epochs):
#                     epoch_data = data[one_sample_size*epoch:one_sample_size*(epoch+1), ch]

#                     # normalization
#                     norm = np.linalg.norm(epoch_data)
#                     epoch_data = epoch_data/norm

#                     # CWT
#                     coef, freqs = pywt.cwt(epoch_data, scales, 'morl')

#                     # plot_scalogram(coef, n_samples=samples, n_scale=fNIRs_scale)
#                     total_epochs_cwt.append(coef)
#                     # print(coef.shape) # (fNIRs_scale, one_sample_size) = (70, 2500)
#                 total_epochs_cwt = np.stack(total_epochs_cwt, axis=0)
#                 acwt = np.mean(total_epochs_cwt, axis=0)
#                 channel_acwts.append(acwt)
            
#             this_file_acwt = np.stack(channel_acwts, axis=0) # (32, 70, 2500)
#             patients.append(this_file_acwt)
              
#         result_feat = np.stack(patients, axis=0)
#         print(level, '->', result_feat.shape) # (patient num, 32, 70, 2500)

#         save_path = save_root + 'Method3/'
#         if not os.path.exists(save_path):
#             os.makedirs(save_path)
#         np.save(save_path + level + '_input.npy', result_feat)


if __name__ == "__main__":

    Method1()
    # Method2()
    # Method3()
import pandas as pd
import numpy as np
import scipy
import pickle
import spkit as sp
import matplotlib.pyplot as plt

def plot():
    plt.figure(figsize=(13,6))
    plt.subplot(211)
    plt.plot(t,x)
    plt.xlim([t[0],t[-1]])
    plt.subplot(212)
    plt.imshow(abs(XW),aspect='auto',origin ='lower', cmap=plt.cm.jet, extent=[t[0], t[-1], S[0], S[-1]],interpolation=interpolation )
    #plt.subplot(313)
    #plt.imshow(np.angle(XW),aspect='auto',origin ='lower', cmap=plt.cm.jet, extent=[t[0], t[-1], S[0], S[-1]],interpolation='sinc' )
    plt.ylabel('scale')
    plt.xlabel('time')
    plt.show()

def EEG_Scalogram_sliced():

    eeg_sampling_rate = 500
    slice_sec = 5
    sample_len = eeg_sampling_rate * slice_sec
    for level_data in all_levels: # ad 26 -> norm 64 -> mci 46
        for patient in level_data: # 26
            for act_num, act_data in enumerate(patient): # 6 = RO, C1, C2, N1, N2, V
                if act_num == 0: # RO
                    data = act_data[eeg_sampling_rate*60:, :] # RO - 60 sec
                    # print(data.shape) # (30000, 32)
                    epochs = int(data.shape[0] / sample_len) # 30000 / 2500 = 12
                    eeg_channels = data.shape[1] # 32
                    for ch in range(eeg_channels): # 0~31
                        one_ch_data = data[:, ch] # (30000,)
                        for epoch in range(epochs):
                            one_ch_data_sample = one_ch_data[sample_len*epoch:sample_len*(epoch+1)] # (2500,)
                            t = np.arange(len(one_ch_data_sample)) / eeg_sampling_rate

                            # Return: XW = (len(scales), len(data_time_len)) = (5, 2500)
                            XW, scales = sp.cwt.ScalogramCWT(one_ch_data_sample, t, fs=eeg_sampling_rate, wType='cMaxican', PlotW=False, PlotPSD=True)


                else:
                    data_list = act_data
                    for data_sample in data_list: # 5 sec (2500, 32)
                        # print(data_sample.shape) # (2500, 32)
                        for ch in range(eeg_channels): # 0~31
                            one_ch_data_sample = data_sample[:, ch] # (2500,)
                            t = np.arange(len(one_ch_data_sample)) / eeg_sampling_rate
                            
                            # Return: XW = (len(scales), len(data_time_len)) = (5, 2500)
                            XW, scales = sp.cwt.ScalogramCWT(one_ch_data_sample, t, fs=eeg_sampling_rate, wType='cMaxican', PlotW=False, PlotPSD=True)

                            plot()


if __name__ == "__main__":

    sliced_pth = './pickles/only_eeg_sliced_data.pkl'

    with open(sliced_pth, 'rb') as f:
        loaded_data = pickle.load(f) # ad, cn, ns, pre order
        print('>> only_eeg_sliced_data.pkl loaded.')

    levels = ['AD', 'NORMAL', 'NP', 'MCI']
    ad = loaded_data[0]['EEG']
    norm = loaded_data[1]['EEG']
    mci = loaded_data[3]['EEG']
    print(len(ad), len(norm), len(mci))
    all_levels = [ad, norm, mci]

    # sp.eeg.eeg_processing.RhythmicDecomposition(E, fs,)
    EEG_Scalogram_sliced()


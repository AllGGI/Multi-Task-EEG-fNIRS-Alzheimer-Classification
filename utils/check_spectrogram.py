import pandas as pd
import numpy as np
import scipy
import pickle
import spkit as sp
import os

import matplotlib
import matplotlib.pyplot as plt
# from matplotlib.pyplot import specgram
from scipy.signal import spectrogram
from scipy.signal import butter, filtfilt



def plot(data, n_seg, n_overlap, save_name):

    f, tt, Sxx = spectrogram(data, fs=500, nperseg=n_seg, noverlap=n_overlap)

    plt.pcolormesh(tt, f, Sxx, shading='gouraud')
    plt.title('Spectrogram')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.ylim(0, 40)
    # plt.show()

    plt.savefig(save_name)

    return Sxx


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y




def EEG_RhythmicDecomposition_sliced(alllevel_eeg_data, slice_sec):
    '''
        Decompose EEG Signal(s)-all the channels in Rhythms and compute power in each band for each channel
        --------------------------------------------------------------------------------------------------
        # Delta, Theta, Alpha, Beta, Gamma1, Gamma2
        default band = [[4],[4,8],[8,14],[14,30],[30,47],[47]]

        input
        E: EEG segment of shape (n,nch)
        fs: sampling rate
        fBands: list of frequency bands - if None: fBands =[[4],[4,8],[8,14],[14,30],[30,47],[47]]

        output
        Px: sum of the power in a band  -  shape (number of bands,nch)
        Pm: mean power in a band       -  shape (number of bands,nch)
        Pd: standard deviation of power in a band  -  shape (number of bands,nch)
    '''
    
    ###################################################################################
    eeg_sampling_rate = 500
    eeg_channels = 32
    eeg_bands = [[4],[4,8],[8,14],[14,30],[30,50],[50]]

    for i, level_data in enumerate(alllevel_eeg_data): # ad 26 -> norm 64 -> mci 46
        patients = []
        for p_idx, patient in enumerate(level_data): # 26
            act_out = [[] for i in range(6)]

            if not os.path.exists('./spec_fig/level-'+str(i)+'/' + str(p_idx) + '/'):
                os.makedirs('./spec_fig/level-'+str(i)+'/' + str(p_idx) + '/')

            for act_num, act_data in enumerate(patient): # 6 = RO, C1, C2, N1, N2, V
                if act_num == 0: # RO (30000=[0,30 sec], 32)
                    data = act_data[:eeg_sampling_rate*60, :] # RO - 60 sec (30000,32)
                    # print(data.shape) # (30000, 32)
                    n_seg = 512
                    n_overlap = 256

                    for ch in range(32):
                        ch_sample = data[:,ch]
                        bandpassed = butter_bandpass_filter(ch_sample, lowcut=4, highcut=40, fs=500, order=5)
                        plot(bandpassed, n_seg, n_overlap, save_name='./spec_fig/level-'+str(i)+'/' + str(p_idx) + '/' + str(act_num) + '_ch-' + str(ch) + '.png')


                if act_num == 1 or act_num == 2 or act_num == 3 or act_num == 4:
                    data = act_data
                    n_seg = 256
                    n_overlap = 128
                    # for data_sample in act_data: # (1500, 32)
                    #     for ch in range(32):
                    #         ch_sample = data_sample[:,ch]
                    #         bandpassed = butter_bandpass_filter(ch_sample, lowcut=4, highcut=40, fs=500, order=5)
                    #         plot(bandpassed)
                    #         raise
                    for ch in range(32):
                        ch_sample = data[:,ch]
                        bandpassed = butter_bandpass_filter(ch_sample, lowcut=4, highcut=40, fs=500, order=5)
                        plot(bandpassed, n_seg, n_overlap, save_name='./fig/level-'+str(i)+'/' + str(p_idx) + '/' + str(act_num) + '_ch-' + str(ch) + '.png')
                        

                elif act_num == 5: # Verbal (15000=[0,30 sec], 32)
                    data_list = act_data

                    n_seg = 512
                    n_overlap = 256
                    for data_idx, data_sample in enumerate(data_list): # 30 sec (15000, 32) * 6 segs
                        # print(data_sample.shape) # (15000, 32)
                        for ch in range(32):
                            ch_sample = data_sample[:,ch]
                            bandpassed = butter_bandpass_filter(ch_sample, lowcut=4, highcut=40, fs=500, order=5)
                            plot(bandpassed, n_seg, n_overlap, save_name='./fig/level-'+str(i)+'/' + str(p_idx) + '/' + str(act_num) + '_' + str(data_idx) + '_ch-' + str(ch) + '.png')



            break

    ###################################################################################
    

    

def FNIRS_RhythmicDecomposition_sliced(alllevel_fnirs_data, slice_sec):

    ###################################################################################
    fnirs_sampling_rate = 8
    # slice_sec = 5
    # sample_len = fnirs_sampling_rate * slice_sec
    # fnirs_channels = 6
    fnirs_band = [[0.01, 0.04], [0.04, 0.15]]

    for i, level_data in enumerate(alllevel_fnirs_data): # ad 26 -> norm 64 -> mci 46
        if i != 0: continue 

        patients = []
        for patient_idx, patient in enumerate(level_data): # 26
            act_out = [[] for i in range(6)]
            for act_num, act_data in enumerate(patient): # 6 = RO, C1, C2, N1, N2, V
                if act_num == 0: # RO
                    hb_data = act_data['Hb'][:fnirs_sampling_rate*60, :] # RO - 60 sec (480, 6)
                    hbo_data = act_data['HbO'][:fnirs_sampling_rate*60, :] # RO - 60 sec (480, 6)
                    thb_data = act_data['THb'][:fnirs_sampling_rate*60, :] # RO - 60 sec (480, 6)

                    n_seg = 40
                    n_overlap = 20
                    



                elif act_num == 3: # C, N (24=[-1,2 sec], 6)
                    n_seg = 16
                    n_overlap = 8
                    hb_act_data = act_data['Hb']
                    hbo_act_data = act_data['HbO']
                    thb_act_data = act_data['THb']

                    # for hb_data_sample, hbo_data_sample, thb_data_sample in zip(hb_act_data, hbo_act_data, thb_act_data): # dict
                
                    #     plt.title('patient-' + str(patient_idx) + '_Hb-act:'+str(act_num))
                    #     plt.plot(np.arange(len(hb_data_sample)), hb_data_sample)
                    #     plt.legend(['ch-' + str(ch) for ch in range(6)])
                    #     plt.savefig('./fig/' + 'patient-' + str(patient_idx) + '_Hb-act-' + str(act_num) + '.png')
                    #     plt.show()
                        
                    #     plt.title('patient-' + str(patient_idx) + '_HbO-act:'+str(act_num))
                    #     plt.plot(np.arange(len(hbo_data_sample)), hbo_data_sample)
                    #     plt.legend(['ch-' + str(ch) for ch in range(6)])
                    #     plt.savefig('./fig/' + 'patient-' + str(patient_idx) + '_HbO-act-' + str(act_num) + '.png')
                    #     plt.show()
                                    
                    #     plt.title('patient-' + str(patient_idx) + '_THb-act:'+str(act_num))
                    #     plt.plot(np.arange(len(thb_data_sample)), thb_data_sample)
                    #     plt.legend(['ch-' + str(ch) for ch in range(6)])
                    #     plt.savefig('./fig/' + 'patient-' + str(patient_idx) + '_THb-act-' + str(act_num) + '.png')
                    #     plt.show()
                        

                    #     break
                        


                elif act_num == 5: # V (240=30 sec, 6)
                    n_seg = 40
                    n_overlap = 20
                    hb_act_data = act_data['Hb']
                    hbo_act_data = act_data['HbO']
                    thb_act_data = act_data['THb']
                    # print(len(hb_act_data), len(hbo_act_data), len(thb_act_data)) # each (240, 6)

                    for hb_data_sample, hbo_data_sample, thb_data_sample in zip(hb_act_data, hbo_act_data, thb_act_data): # dict
                        
                        hb_data_sample = [butter_bandpass_filter(hb_data_sample[:,ch], 0.01, 0.15, 8, 5) for ch in range(6)]
                        hb_data_sample = np.stack(hb_data_sample, axis=-1)

                        plt.title('patient-' + str(patient_idx) + '_Hb-act:'+str(act_num))
                        plt.plot(np.arange(len(hb_data_sample)), hb_data_sample)
                        plt.legend(['ch-' + str(ch) for ch in range(6)])
                        # plt.savefig('./fig/' + 'patient-' + str(patient_idx) + '_Hb-act-' + str(act_num) + '.png')
                        plt.show()

                        hbo_data_sample = [butter_bandpass_filter(hbo_data_sample[:,ch], 0.01, 0.15, 8, 5) for ch in range(6)]
                        hbo_data_sample = np.stack(hbo_data_sample, axis=-1)

                        plt.title('patient-' + str(patient_idx) + '_HbO-act:'+str(act_num))
                        plt.plot(np.arange(len(hbo_data_sample)), hbo_data_sample)
                        plt.legend(['ch-' + str(ch) for ch in range(6)])
                        # plt.savefig('./fig/' + 'patient-' + str(patient_idx) + '_HbO-act-' + str(act_num) + '.png')
                        plt.show()

                        thb_data_sample = [butter_bandpass_filter(thb_data_sample[:,ch], 0.01, 0.15, 8, 5) for ch in range(6)]
                        thb_data_sample = np.stack(thb_data_sample, axis=-1)
  
                        plt.title('patient-' + str(patient_idx) + '_THb-act:'+str(act_num))
                        plt.plot(np.arange(len(thb_data_sample)), thb_data_sample)
                        plt.legend(['ch-' + str(ch) for ch in range(6)])
                        # plt.savefig('./fig/' + 'patient-' + str(patient_idx) + '_THb-act-' + str(act_num) + '.png')
                        plt.show()
                        

                        break

    ###################################################################################




if __name__ == "__main__":
    slice_sec_num = 5

    sliced_pth = './pickles/eeg-' + str(slice_sec_num) + 'sec_fnirs-full_baselinecorrected_fnirs_sliced.pkl'
    new_AD_sliced_pth = './pickles/new-AD_eeg-' + str(slice_sec_num) + 'sec_fnirs-full_baselinecorrected_fnirs_sliced.pkl'
    # sliced_pth = './pickles/eeg_full_fnirs_full_baselinecorrected.pkl'
    # new_AD_sliced_pth = './pickles/new-AD_eeg_full_fnirs_full_baselinecorrected.pkl'
    
    sliced = True
    with_fnirs = True


    if sliced:

        with open(sliced_pth, 'rb') as f:
            loaded_data = pickle.load(f) # ad, cn, ns, pre order
            print('>> ' + sliced_pth + ' loaded.')

        levels = ['AD', 'NORMAL', 'MCI']
        ad_eeg = loaded_data[0]['EEG']
        norm_eeg = loaded_data[1]['EEG']
        mci_eeg = loaded_data[3]['EEG']

        ad_fnirs = loaded_data[0]['fNIRs']
        norm_fnirs = loaded_data[1]['fNIRs']
        mci_fnirs = loaded_data[3]['fNIRs']


        with open(new_AD_sliced_pth, 'rb') as f:
            new_loaded_data = pickle.load(f) # ad, cn, ns, pre order
            print('>> ' + new_AD_sliced_pth + ' loaded.')    
        ad_eeg += new_loaded_data[0]['EEG']
        ad_fnirs += new_loaded_data[0]['fNIRs']

        print(len(ad_eeg), len(norm_eeg), len(mci_eeg))
        all_levels_eeg = [ad_eeg, norm_eeg, mci_eeg]
        print(len(ad_fnirs), len(norm_fnirs), len(mci_fnirs))
        all_levels_fnirs = [ad_fnirs, norm_fnirs, mci_fnirs]

        # EEG_RhythmicDecomposition_sliced(all_levels_eeg, slice_sec=slice_sec_num)
        FNIRS_RhythmicDecomposition_sliced(all_levels_fnirs, slice_sec=slice_sec_num)


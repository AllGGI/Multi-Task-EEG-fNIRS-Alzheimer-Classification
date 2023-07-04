import scipy
from scipy import io
import os
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import pickle
import pandas as pd
from tqdm import tqdm
import copy

# fft function (for fnirs)
def fft(fs, data, plot=False):
    # Get real amplitudes of FFT (only in postive frequencies)
    fft_vals = np.absolute(np.fft.rfft(data))

    # Get frequencies for amplitudes in Hz
    fft_freq = np.fft.rfftfreq(len(data), 1.0/fs)
    
    # Define bands
    # freq_bands = [[0.2+i*0.2,0.4+i*0.2] for i in range(18)]
    # fnirs_bands = {
    #     'VLFO':(0.01, 0.04),
    #     'LFO':(0.04, 0.15),
    # }
    fnirs_bands = {'fnirs':(0.009, 0.08)}
    
    freq_band_fft = []
    for band in fnirs_bands:
        freq_ix = np.where((fft_freq >= fnirs_bands[band][0]) & 
                           (fft_freq < fnirs_bands[band][1]))[0]
        freq_band_fft.append(np.mean(fft_vals[freq_ix]))

    # Plot the data (using pandas here cause it's easy)
    if plot:
        plt.plot(np.array(fnirs_bands)[:,0], freq_band_fft)
        plt.xlabel('Frequency')
        plt.ylabel('Amplitude')
        plt.show()
    return freq_band_fft


def fft_fnirs(levels_list):
    
    fnirs_sampling_rate = 8
    # fnirs_band = [[0.01, 0.1]]
    fnirs_ch = 6

    patient_all_data_normalized = []
    for label, level_data in enumerate(levels_list): # ad 26 -> norm 64 -> mci 46
        for patient in level_data: # 26
            act_out = [[] for i in range(6)]
            for act_num, act_data in enumerate(patient): # 6 = RO, C1, C2, N1, N2, V
                if act_num == 0: # RO
                    hb_data = act_data['Hb'][:fnirs_sampling_rate*60, :] # RO - 60 sec (480, 6)
                    hbo_data = act_data['HbO'][:fnirs_sampling_rate*60, :] # RO - 60 sec (480, 6)
                    thb_data = act_data['THb'][:fnirs_sampling_rate*60, :] # RO - 60 sec (480, 6)

                elif act_num == 1 or act_num == 2 or act_num == 3 or act_num == 4: # C, N (24=[-1,2 sec], 6)
                    hb_data = act_data['Hb'] # 3 sec (24, 6)
                    hbo_data = act_data['HbO'] # 3 sec (24, 6)
                    thb_data = act_data['THb'] # 3 sec (24, 6)

                elif act_num == 5: # V (240=30 sec, 6)
                    hb_data = act_data['Hb'] # 30 sec (240, 6)
                    hbo_data = act_data['HbO'] # 30 sec (240, 6)
                    thb_data = act_data['THb'] # 30 sec (240, 6)


                fnirs_allch = []
                # fft fNIRS for 6 channel
                for ch in range(fnirs_ch):
                    hb = hb_data[:,ch]
                    hbo = hbo_data[:,ch]
                    thb = thb_data[:,ch]
                    
                    hb_fft = fft(fnirs_sampling_rate, hb)
                    hbo_fft = fft(fnirs_sampling_rate, hbo)
                    thb_fft = fft(fnirs_sampling_rate, thb)
                    
                    # normalize
                    norm = np.linalg.norm(hb_fft)
                    hb_fft = hb_fft/norm
                    norm = np.linalg.norm(hbo_fft)
                    hbo_fft = hbo_fft/norm
                    norm = np.linalg.norm(thb_fft)
                    thb_fft = thb_fft/norm

                    fnirs_allch.append([hb_fft.tolist(), hbo_fft.tolist(), thb_fft.tolist()])

                act_out[act_num] = fnirs_allch
            patient_all_data_normalized.append([label, act_out])


    # !!! POSTPROCESSING !!! : Flatten 'fft_data_norm' for dataframe
    # patient_all_data_normalized[{patient_num}][{0:label, 1:data}][{act_num}][{fnirs}] 
    flattened_norm_data = []
    for i in range(len(patient_all_data_normalized)):
        label = patient_all_data_normalized[i][0]
        one_patient_data = patient_all_data_normalized[i][1]
        all_fnirs_data = []
        for act_num in range(6):
            fnirs_data = one_patient_data[act_num]

            # [[2 band] * 3 hb] * 6ch
            all_fnirs_data += [fnirs_data[ch][hb_type][band] for ch in range(fnirs_ch) for hb_type in range(3) for band in range(2)] # fnirs 2 bands

        flattened_norm_data.append([label] + all_fnirs_data)

    return flattened_norm_data # flattened 'patient_all_data_normalized'


if __name__ == "__main__":
    
    sliced_pth = './pickles/eeg-3sec_fnirs-full_baselinecorrected.pkl'
    new_AD_sliced_pth = './pickles/new-AD_eeg-3sec_fnirs-full_baselinecorrected.pkl'
    sliced = True
    with_fnirs = True


    if sliced:

        # if not os.path.exists('./more_AD_power_npy/sliced/'):
        #     os.makedirs('./more_AD_power_npy/sliced/')

        with open(sliced_pth, 'rb') as f:
            loaded_data = pickle.load(f) # ad, cn, ns, pre order
            print('>> ' + sliced_pth + ' loaded.')

        levels = ['AD', 'NORMAL', 'MCI']
        # ad_eeg = loaded_data[0]['EEG']
        # norm_eeg = loaded_data[1]['EEG']
        # mci_eeg = loaded_data[3]['EEG']

        ad_fnirs = loaded_data[0]['fNIRs']
        norm_fnirs = loaded_data[1]['fNIRs']
        mci_fnirs = loaded_data[3]['fNIRs']


        with open(new_AD_sliced_pth, 'rb') as f:
            new_loaded_data = pickle.load(f) # ad, cn, ns, pre order
            print('>> ' + new_AD_sliced_pth + ' loaded.')    
        # ad_eeg += new_loaded_data[0]['EEG']
        ad_fnirs += new_loaded_data[0]['fNIRs']

        # print(len(ad_eeg), len(norm_eeg), len(mci_eeg))
        # all_levels_eeg = [ad_eeg, norm_eeg, mci_eeg]

        print(len(ad_fnirs), len(norm_fnirs), len(mci_fnirs))
        all_levels_fnirs = [ad_fnirs, norm_fnirs, mci_fnirs]

        flatten_norm_fnirs = fft_fnirs(all_levels_fnirs)

        ####################################################################################################
        # 5) making column
        column = ['label']
        fnirs_band_names = ['vlfo', 'lfo']
        fnirs_props = ['Hb', 'HbO','THb']

        for act_num in range(6):
            for ch in range(6):
                for prop in fnirs_props:
                    for band in fnirs_band_names: # 2 bands
                        column += ['fnirs_act-{}_ch-{}_prop-{}_band-{}'.format(act_num, ch, prop, band)]

        ####################################################################################################
        # 6) save csv
        # make save folder
        if not os.path.exists('./final_csv'):
            os.makedirs('./final_csv')
        df = pd.DataFrame(flatten_norm_fnirs, columns=column)
        df.to_csv('./final_csv/fft_data_fnirs.csv', sep=',')
        print('>> csv saved.')

        print(df)


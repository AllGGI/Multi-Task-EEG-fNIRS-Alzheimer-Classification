import pandas as pd
import numpy as np
import scipy
import pickle
import spkit as sp
import os


def EEG_RhythmicDecomposition_sliced(levels, alllevel_eeg_data, slice_sec):
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
    eeg_bands = [[4],[4,8],[8,14],[14,30],[30,40],[40,50]] # [[4],[4,7],[8,10],[10,13],[14,25],[26,40]]

    for i, level_data in enumerate(alllevel_eeg_data): # ad 26 -> norm 64 -> mci 46
        patients = []
        for patient in level_data: # 26
            act_out = [[] for i in range(6)]
            for act_num, act_data in enumerate(patient): # 6 = RO, C1, C2, N1, N2, V
                if act_num == 0: # RO (30000=[0,30 sec], 32)
                    data = act_data[:eeg_sampling_rate*60, :] # RO - 60 sec (30000,32)
                    # print(data.shape) # (30000, 32)
                    n_seg = 512
                    n_overlap = 256
                    Px,Pm,Pd = sp.eeg.eeg_processing.RhythmicDecomposition(E=data, fBands=eeg_bands, fs=eeg_sampling_rate, nperseg=n_seg, noverlap=n_overlap, Sum=True, Mean=True, SD=True)
                    # print(Px.shape, Pm.shape, Pd.shape) # (6 band, 32 ch) each
                    powers = np.stack([Px,Pm,Pd], axis=0)
                    # print(powers.shape)
                    
                    act_out[act_num] = powers # (3,6,32)

                elif act_num == 1 or act_num == 2 or act_num == 3 or act_num == 4: # C, N (1500=[-1,2 sec], 32)
                    data_list = act_data
                    act_out_1_sample_Px, act_out_1_sample_Pm, act_out_1_sample_Pd = [], [], []

                    n_seg = 256
                    n_overlap = 128
                    for data_sample in data_list: # 3 sec (1500, 32)
                        # print(data_sample.shape) # (1500, 32)
                        Px,Pm,Pd = sp.eeg.eeg_processing.RhythmicDecomposition(E=data_sample, fBands=eeg_bands, fs=eeg_sampling_rate, nperseg=n_seg, noverlap=n_overlap, Sum=True, Mean=True, SD=True)
                        # print(Px.shape, Pm.shape, Pd.shape) # (6 band, 32 ch) each

                        # powers = np.stack([Px,Pm,Pd], axis=0)
                        # print(powers.shape)
                        act_out_1_sample_Px.append(Px) # (6,32)
                        act_out_1_sample_Pm.append(Pm) # (6,32)
                        act_out_1_sample_Pd.append(Pd) # (6,32)
                    
                    act_out_Px = np.mean(np.stack(act_out_1_sample_Px, axis=0), axis=0) # (6,32)
                    act_out_Pm = np.mean(np.stack(act_out_1_sample_Pm, axis=0), axis=0)
                    act_out_Pd = np.mean(np.stack(act_out_1_sample_Pd, axis=0), axis=0)
                    this_act_out = np.stack([act_out_Px, act_out_Pm, act_out_Pd], axis=0) # (3,6,32)

                    act_out[act_num] = this_act_out # (3,6,32)

                elif act_num == 5: # Verbal (15000=[0,30 sec], 32)

                    data_list = act_data
                    act_out_1_sample_Px, act_out_1_sample_Pm, act_out_1_sample_Pd = [], [], []

                    n_seg = 512
                    n_overlap = 256
                    for data_sample in data_list: # 30 sec (15000, 32) * 6 segs
                        # print(data_sample.shape) # (15000, 32)
                        
                        Px,Pm,Pd = sp.eeg.eeg_processing.RhythmicDecomposition(E=data_sample, fBands=eeg_bands, fs=eeg_sampling_rate, nperseg=n_seg, noverlap=n_overlap, Sum=True, Mean=True, SD=True)
                        # print(Px.shape, Pm.shape, Pd.shape) # (6 band, 32 ch) each

                        powers = np.stack([Px,Pm,Pd], axis=0)
                        # print(powers.shape)
                        act_out_1_sample_Px.append(Px) # (6,32)
                        act_out_1_sample_Pm.append(Pm) # (6,32)
                        act_out_1_sample_Pd.append(Pd) # (6,32)

                    act_out_Px = np.mean(np.stack(act_out_1_sample_Px, axis=0), axis=0) # (6,32)
                    act_out_Pm = np.mean(np.stack(act_out_1_sample_Pm, axis=0), axis=0)
                    act_out_Pd = np.mean(np.stack(act_out_1_sample_Pd, axis=0), axis=0)
                    this_act_out = np.stack([act_out_Px, act_out_Pm, act_out_Pd], axis=0) # (3,6,32)

                    act_out[act_num] = this_act_out # (3,6,32)



            patients.append(act_out)
        
        patients = np.asarray(patients)
        # print(patients.shape) # (26, 6, 3, 6, 32)

        np.save('./inputs/' + str(slice_sec) + '-sec/' + levels[i] + '_eeg_power_SMSD.npy', patients)
        print('>> npy saved.')
    ###################################################################################
    

    

def FNIRS_RhythmicDecomposition_sliced(levels, alllevel_fnirs_data, slice_sec):

    ###################################################################################
    fnirs_sampling_rate = 8
    # slice_sec = 5
    # sample_len = fnirs_sampling_rate * slice_sec
    # fnirs_channels = 6
    fnirs_band = [[0.01, 0.04], [0.04, 0.15]]

    for i, level_data in enumerate(alllevel_fnirs_data): # ad 26 -> norm 64 -> mci 46
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
                    hb_Px,hb_Pm,hb_Pd = sp.eeg.eeg_processing.RhythmicDecomposition(E=hb_data, fs=fnirs_sampling_rate, nperseg=n_seg, fBands=fnirs_band, Sum=True, Mean=True, SD =True)
                    hb_powers = np.stack([hb_Px,hb_Pm,hb_Pd], axis=0)

                    hbo_Px,hbo_Pm,hbo_Pd = sp.eeg.eeg_processing.RhythmicDecomposition(E=hbo_data, fs=fnirs_sampling_rate, nperseg=n_seg, fBands=fnirs_band, Sum=True, Mean=True, SD =True)
                    hbo_powers = np.stack([hbo_Px,hbo_Pm,hbo_Pd], axis=0)

                    thb_Px,thb_Pm,thb_Pd = sp.eeg.eeg_processing.RhythmicDecomposition(E=thb_data, fs=fnirs_sampling_rate, nperseg=n_seg, fBands=fnirs_band, Sum=True, Mean=True, SD =True)
                    thb_powers = np.stack([thb_Px,thb_Pm,thb_Pd], axis=0)

                    # print(hb_powers.shape) # (3,4,6)
                    powers = np.stack([hb_powers, hbo_powers, thb_powers], axis=0)
                    act_out[act_num] = powers # (3, 3,4,6)

                elif act_num == 1 or act_num == 2 or act_num == 3 or act_num == 4: # C, N (24=[-1,2 sec], 6)
                    n_seg = 16
                    n_overlap = 8
                    hb_act_data = act_data['Hb']
                    hbo_act_data = act_data['HbO']
                    thb_act_data = act_data['THb']
                    

                    act_out_1_sample_Px, act_out_1_sample_Pm, act_out_1_sample_Pd = {'Hb':[], 'HbO':[], 'THb':[]}, {'Hb':[], 'HbO':[], 'THb':[]}, {'Hb':[], 'HbO':[], 'THb':[]}

                    for hb_data_sample, hbo_data_sample, thb_data_sample in zip(hb_act_data, hbo_act_data, thb_act_data): # dict
                        hb_Px,hb_Pm,hb_Pd = sp.eeg.eeg_processing.RhythmicDecomposition(E=hb_data_sample, fs=fnirs_sampling_rate, nperseg=n_seg, fBands=fnirs_band, Sum=True, Mean=True, SD =True)
                        hbo_Px,hbo_Pm,hbo_Pd = sp.eeg.eeg_processing.RhythmicDecomposition(E=hbo_data_sample, fs=fnirs_sampling_rate, nperseg=n_seg, fBands=fnirs_band, Sum=True, Mean=True, SD =True)
                        thb_Px,thb_Pm,thb_Pd = sp.eeg.eeg_processing.RhythmicDecomposition(E=thb_data_sample, fs=fnirs_sampling_rate, nperseg=n_seg, fBands=fnirs_band, Sum=True, Mean=True, SD =True)
                        
                    
                        act_out_1_sample_Px['Hb'].append(hb_Px)
                        act_out_1_sample_Px['HbO'].append(hbo_Px)
                        act_out_1_sample_Px['THb'].append(thb_Px)

                        act_out_1_sample_Pm['Hb'].append(hb_Pm)
                        act_out_1_sample_Pm['HbO'].append(hbo_Pm)
                        act_out_1_sample_Pm['THb'].append(thb_Pm)

                        act_out_1_sample_Pd['Hb'].append(hb_Pd)
                        act_out_1_sample_Pd['HbO'].append(hbo_Pd)
                        act_out_1_sample_Pd['THb'].append(thb_Pd)

                    
                    hb_act_out_Px = np.mean(np.stack(act_out_1_sample_Px['Hb'], axis=0), axis=0) # (4,6)
                    hb_act_out_Pm = np.mean(np.stack(act_out_1_sample_Pm['Hb'], axis=0), axis=0)
                    hb_act_out_Pd = np.mean(np.stack(act_out_1_sample_Pd['Hb'], axis=0), axis=0)
                    hb_powers = np.stack([hb_act_out_Px, hb_act_out_Pm, hb_act_out_Pd], axis=0) # (3,4,6)
                    

                    hbo_act_out_Px = np.mean(np.stack(act_out_1_sample_Px['HbO'], axis=0), axis=0) # (4,6)
                    hbo_act_out_Pm = np.mean(np.stack(act_out_1_sample_Pm['HbO'], axis=0), axis=0)
                    hbo_act_out_Pd = np.mean(np.stack(act_out_1_sample_Pd['HbO'], axis=0), axis=0)
                    hbo_powers = np.stack([hbo_act_out_Px, hbo_act_out_Pm, hbo_act_out_Pd], axis=0) # (3,4,6)

                    thb_act_out_Px = np.mean(np.stack(act_out_1_sample_Px['THb'], axis=0), axis=0) # (4,6)
                    thb_act_out_Pm = np.mean(np.stack(act_out_1_sample_Pm['THb'], axis=0), axis=0)
                    thb_act_out_Pd = np.mean(np.stack(act_out_1_sample_Pd['THb'], axis=0), axis=0)
                    thb_powers = np.stack([thb_act_out_Px, thb_act_out_Pm, thb_act_out_Pd], axis=0) # (3,4,6)
                    
                    this_act_out = np.stack([hb_powers, hbo_powers, thb_powers], axis=0) # (3hb, 3px, 4band, 6ch)

                    act_out[act_num] = this_act_out # (3hb, 3px, 4band, 6ch)


                elif act_num == 5: # V (240=30 sec, 6)
                    n_seg = 40
                    n_overlap = 20
                    hb_act_data = act_data['Hb']
                    hbo_act_data = act_data['HbO']
                    thb_act_data = act_data['THb']
                    # print(len(hb_act_data), len(hbo_act_data), len(thb_act_data)) # each (240, 6)

                    act_out_1_sample_Px, act_out_1_sample_Pm, act_out_1_sample_Pd = {'Hb':[], 'HbO':[], 'THb':[]}, {'Hb':[], 'HbO':[], 'THb':[]}, {'Hb':[], 'HbO':[], 'THb':[]}

                    for hb_data_sample, hbo_data_sample, thb_data_sample in zip(hb_act_data, hbo_act_data, thb_act_data): # dict
                        
                        hb_Px,hb_Pm,hb_Pd = sp.eeg.eeg_processing.RhythmicDecomposition(E=hb_data_sample, fs=fnirs_sampling_rate, nperseg=n_seg, noverlap=n_overlap, fBands=fnirs_band, Sum=True, Mean=True, SD =True)
                        hbo_Px,hbo_Pm,hbo_Pd = sp.eeg.eeg_processing.RhythmicDecomposition(E=hbo_data_sample, fs=fnirs_sampling_rate, nperseg=n_seg, noverlap=n_overlap, fBands=fnirs_band, Sum=True, Mean=True, SD =True)
                        thb_Px,thb_Pm,thb_Pd = sp.eeg.eeg_processing.RhythmicDecomposition(E=thb_data_sample, fs=fnirs_sampling_rate, nperseg=n_seg, noverlap=n_overlap, fBands=fnirs_band, Sum=True, Mean=True, SD =True)

                        # print(hb_Px.shape, hb_Pm.shape, hb_Pd.shape) # (4,6) each

                        act_out_1_sample_Px['Hb'].append(hb_Px)
                        act_out_1_sample_Px['HbO'].append(hbo_Px)
                        act_out_1_sample_Px['THb'].append(thb_Px)

                        act_out_1_sample_Pm['Hb'].append(hb_Pm)
                        act_out_1_sample_Pm['HbO'].append(hbo_Pm)
                        act_out_1_sample_Pm['THb'].append(thb_Pm)

                        act_out_1_sample_Pd['Hb'].append(hb_Pd)
                        act_out_1_sample_Pd['HbO'].append(hbo_Pd)
                        act_out_1_sample_Pd['THb'].append(thb_Pd)

                    hb_act_out_Px = np.mean(np.stack(act_out_1_sample_Px['Hb'], axis=0), axis=0) # (4,6)
                    hb_act_out_Pm = np.mean(np.stack(act_out_1_sample_Pm['Hb'], axis=0), axis=0)
                    hb_act_out_Pd = np.mean(np.stack(act_out_1_sample_Pd['Hb'], axis=0), axis=0)
                    hb_powers = np.stack([hb_act_out_Px, hb_act_out_Pm, hb_act_out_Pd], axis=0) # (3,4,6)

                    hbo_act_out_Px = np.mean(np.stack(act_out_1_sample_Px['HbO'], axis=0), axis=0) # (4,6)
                    hbo_act_out_Pm = np.mean(np.stack(act_out_1_sample_Pm['HbO'], axis=0), axis=0)
                    hbo_act_out_Pd = np.mean(np.stack(act_out_1_sample_Pd['HbO'], axis=0), axis=0)
                    hbo_powers = np.stack([hbo_act_out_Px, hbo_act_out_Pm, hbo_act_out_Pd], axis=0) # (3,4,6)

                    thb_act_out_Px = np.mean(np.stack(act_out_1_sample_Px['THb'], axis=0), axis=0) # (4,6)
                    thb_act_out_Pm = np.mean(np.stack(act_out_1_sample_Pm['THb'], axis=0), axis=0)
                    thb_act_out_Pd = np.mean(np.stack(act_out_1_sample_Pd['THb'], axis=0), axis=0)
                    thb_powers = np.stack([thb_act_out_Px, thb_act_out_Pm, thb_act_out_Pd], axis=0) # (3,4,6)
                    
                    this_act_out = np.stack([hb_powers, hbo_powers, thb_powers], axis=0) # (3hb, 3px, 4band, 6ch)

                    act_out[act_num] = this_act_out # (3hb, 3px, 4band, 6ch)



            patients.append(act_out)
        
        patients = np.asarray(patients)
        print(patients.shape) # (26, 6, 3, 3, 4, 6)

        np.save('./inputs/' + str(slice_sec) + '-sec/' + levels[i] + '_fnirs_power_SMSD.npy', patients)
        print('>> npy saved.')
    ###################################################################################



def EEG_RhythmicDecomposition_full(levels, alllevel_eeg_data):
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
    eeg_bands = [[4],[4,8],[8,14],[14,30],[30,40],[40,50]] # [[4],[4,7],[8,10],[10,13],[14,25],[26,40]]

    for i, level_data in enumerate(alllevel_eeg_data): # ad 26 -> norm 64 -> mci 46
        patients = []
        for patient in level_data: # 26
            act_out = [[] for i in range(6)]
            for act_num, act_data in enumerate(patient): # 6 = RO, C1, C2, N1, N2, V
                if act_num == 0: # RO (30000=[0,30 sec], 32)
                    data = act_data[:eeg_sampling_rate*60, :] # RO - 60 sec (30000,32)
                    # print(data.shape) # (30000, 32)
                    n_seg = 512
                    n_overlap = 256
                    Px,Pm,Pd = sp.eeg.eeg_processing.RhythmicDecomposition(E=data, fBands=eeg_bands, fs=eeg_sampling_rate, nperseg=n_seg, noverlap=n_overlap, Sum=True, Mean=True, SD=True)
                    # print(Px.shape, Pm.shape, Pd.shape) # (6 band, 32 ch) each
                    powers = np.stack([Px,Pm,Pd], axis=0)
                    # print(powers.shape)
                    
                    act_out[act_num] = powers # (3,6,32)

                elif act_num == 1 or act_num == 2 or act_num == 3 or act_num == 4: # C, N (100000, 32)
                    n_seg = 256
                    n_overlap = 128

                    data = act_data # (100000, 32)
                    Px,Pm,Pd = sp.eeg.eeg_processing.RhythmicDecomposition(E=data, fBands=eeg_bands, fs=eeg_sampling_rate, nperseg=n_seg, noverlap=n_overlap, Sum=True, Mean=True, SD=True)
                    powers = np.stack([Px,Pm,Pd], axis=0)

                    act_out[act_num] = powers # (3,6,32)

                elif act_num == 5: # Verbal (15000=[0,30 sec], 32)

                    data_list = act_data
                    act_out_1_sample_Px, act_out_1_sample_Pm, act_out_1_sample_Pd = [], [], []

                    n_seg = 512
                    n_overlap = 256
                    for data_sample in data_list: # 30 sec (15000, 32) * 6 segs
                        # print(data_sample.shape) # (15000, 32)
                        
                        Px,Pm,Pd = sp.eeg.eeg_processing.RhythmicDecomposition(E=data_sample, fBands=eeg_bands, fs=eeg_sampling_rate, nperseg=n_seg, noverlap=n_overlap, Sum=True, Mean=True, SD=True)
                        # print(Px.shape, Pm.shape, Pd.shape) # (6 band, 32 ch) each

                        powers = np.stack([Px,Pm,Pd], axis=0)
                        # print(powers.shape)
                        act_out_1_sample_Px.append(Px) # (6,32)
                        act_out_1_sample_Pm.append(Pm) # (6,32)
                        act_out_1_sample_Pd.append(Pd) # (6,32)

                    act_out_Px = np.mean(np.stack(act_out_1_sample_Px, axis=0), axis=0) # (6,32)
                    act_out_Pm = np.mean(np.stack(act_out_1_sample_Pm, axis=0), axis=0)
                    act_out_Pd = np.mean(np.stack(act_out_1_sample_Pd, axis=0), axis=0)
                    this_act_out = np.stack([act_out_Px, act_out_Pm, act_out_Pd], axis=0) # (3,6,32)

                    act_out[act_num] = this_act_out # (3,6,32)

            patients.append(act_out)
        
        patients = np.asarray(patients)
        print(patients.shape) # (26, 6, 3, 6, 32)

        np.save('./inputs/full/' + levels[i] + '_eeg_power_SMSD.npy', patients)
        print('>> npy saved.')
    ###################################################################################
    

def FNIRS_RhythmicDecomposition_full(levels, alllevel_fnirs_data):

    ###################################################################################
    fnirs_sampling_rate = 8
    # slice_sec = 5
    # sample_len = fnirs_sampling_rate * slice_sec
    # fnirs_channels = 6
    fnirs_band = [[0.01, 0.04], [0.04, 0.15]]

    for i, level_data in enumerate(alllevel_fnirs_data): # ad 26 -> norm 64 -> mci 46
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
                    hb_Px,hb_Pm,hb_Pd = sp.eeg.eeg_processing.RhythmicDecomposition(E=hb_data, fs=fnirs_sampling_rate, nperseg=n_seg, fBands=fnirs_band, Sum=True, Mean=True, SD =True)
                    hb_powers = np.stack([hb_Px,hb_Pm,hb_Pd], axis=0)

                    hbo_Px,hbo_Pm,hbo_Pd = sp.eeg.eeg_processing.RhythmicDecomposition(E=hbo_data, fs=fnirs_sampling_rate, nperseg=n_seg, fBands=fnirs_band, Sum=True, Mean=True, SD =True)
                    hbo_powers = np.stack([hbo_Px,hbo_Pm,hbo_Pd], axis=0)

                    thb_Px,thb_Pm,thb_Pd = sp.eeg.eeg_processing.RhythmicDecomposition(E=thb_data, fs=fnirs_sampling_rate, nperseg=n_seg, fBands=fnirs_band, Sum=True, Mean=True, SD =True)
                    thb_powers = np.stack([thb_Px,thb_Pm,thb_Pd], axis=0)

                    # print(hb_powers.shape) # (3,4,6)
                    powers = np.stack([hb_powers, hbo_powers, thb_powers], axis=0)
                    act_out[act_num] = powers # (3, 3,4,6)

                elif act_num == 1 or act_num == 2 or act_num == 3 or act_num == 4: # C, N (24=[-1,2 sec], 6)
                    hb_data = act_data['Hb']
                    hbo_data = act_data['HbO']
                    thb_data = act_data['THb']

                    n_seg = 16
                    n_overlap = 8
                    hb_Px,hb_Pm,hb_Pd = sp.eeg.eeg_processing.RhythmicDecomposition(E=hb_data, fs=fnirs_sampling_rate, nperseg=n_seg, fBands=fnirs_band, Sum=True, Mean=True, SD =True)
                    hb_powers = np.stack([hb_Px,hb_Pm,hb_Pd], axis=0)

                    hbo_Px,hbo_Pm,hbo_Pd = sp.eeg.eeg_processing.RhythmicDecomposition(E=hbo_data, fs=fnirs_sampling_rate, nperseg=n_seg, fBands=fnirs_band, Sum=True, Mean=True, SD =True)
                    hbo_powers = np.stack([hbo_Px,hbo_Pm,hbo_Pd], axis=0)

                    thb_Px,thb_Pm,thb_Pd = sp.eeg.eeg_processing.RhythmicDecomposition(E=thb_data, fs=fnirs_sampling_rate, nperseg=n_seg, fBands=fnirs_band, Sum=True, Mean=True, SD =True)
                    thb_powers = np.stack([thb_Px,thb_Pm,thb_Pd], axis=0)

                    # print(hb_powers.shape) # (3,4,6)
                    powers = np.stack([hb_powers, hbo_powers, thb_powers], axis=0)

                    act_out[act_num] = powers # (3hb, 3px, 4band, 6ch)


                elif act_num == 5: # V (240=30 sec, 6)
                    n_seg = 40
                    n_overlap = 20
                    hb_act_data = act_data['Hb']
                    hbo_act_data = act_data['HbO']
                    thb_act_data = act_data['THb']
                    # print(len(hb_act_data), len(hbo_act_data), len(thb_act_data)) # each (240, 6)

                    act_out_1_sample_Px, act_out_1_sample_Pm, act_out_1_sample_Pd = {'Hb':[], 'HbO':[], 'THb':[]}, {'Hb':[], 'HbO':[], 'THb':[]}, {'Hb':[], 'HbO':[], 'THb':[]}

                    for hb_data_sample, hbo_data_sample, thb_data_sample in zip(hb_act_data, hbo_act_data, thb_act_data): # dict
                        
                        hb_Px,hb_Pm,hb_Pd = sp.eeg.eeg_processing.RhythmicDecomposition(E=hb_data_sample, fs=fnirs_sampling_rate, nperseg=n_seg, noverlap=n_overlap, fBands=fnirs_band, Sum=True, Mean=True, SD =True)
                        hbo_Px,hbo_Pm,hbo_Pd = sp.eeg.eeg_processing.RhythmicDecomposition(E=hbo_data_sample, fs=fnirs_sampling_rate, nperseg=n_seg, noverlap=n_overlap, fBands=fnirs_band, Sum=True, Mean=True, SD =True)
                        thb_Px,thb_Pm,thb_Pd = sp.eeg.eeg_processing.RhythmicDecomposition(E=thb_data_sample, fs=fnirs_sampling_rate, nperseg=n_seg, noverlap=n_overlap, fBands=fnirs_band, Sum=True, Mean=True, SD =True)

                        # print(hb_Px.shape, hb_Pm.shape, hb_Pd.shape) # (4,6) each

                        act_out_1_sample_Px['Hb'].append(hb_Px)
                        act_out_1_sample_Px['HbO'].append(hbo_Px)
                        act_out_1_sample_Px['THb'].append(thb_Px)

                        act_out_1_sample_Pm['Hb'].append(hb_Pm)
                        act_out_1_sample_Pm['HbO'].append(hbo_Pm)
                        act_out_1_sample_Pm['THb'].append(thb_Pm)

                        act_out_1_sample_Pd['Hb'].append(hb_Pd)
                        act_out_1_sample_Pd['HbO'].append(hbo_Pd)
                        act_out_1_sample_Pd['THb'].append(thb_Pd)

                    hb_act_out_Px = np.mean(np.stack(act_out_1_sample_Px['Hb'], axis=0), axis=0) # (4,6)
                    hb_act_out_Pm = np.mean(np.stack(act_out_1_sample_Pm['Hb'], axis=0), axis=0)
                    hb_act_out_Pd = np.mean(np.stack(act_out_1_sample_Pd['Hb'], axis=0), axis=0)
                    hb_powers = np.stack([hb_act_out_Px, hb_act_out_Pm, hb_act_out_Pd], axis=0) # (3,4,6)

                    hbo_act_out_Px = np.mean(np.stack(act_out_1_sample_Px['HbO'], axis=0), axis=0) # (4,6)
                    hbo_act_out_Pm = np.mean(np.stack(act_out_1_sample_Pm['HbO'], axis=0), axis=0)
                    hbo_act_out_Pd = np.mean(np.stack(act_out_1_sample_Pd['HbO'], axis=0), axis=0)
                    hbo_powers = np.stack([hbo_act_out_Px, hbo_act_out_Pm, hbo_act_out_Pd], axis=0) # (3,4,6)

                    thb_act_out_Px = np.mean(np.stack(act_out_1_sample_Px['THb'], axis=0), axis=0) # (4,6)
                    thb_act_out_Pm = np.mean(np.stack(act_out_1_sample_Pm['THb'], axis=0), axis=0)
                    thb_act_out_Pd = np.mean(np.stack(act_out_1_sample_Pd['THb'], axis=0), axis=0)
                    thb_powers = np.stack([thb_act_out_Px, thb_act_out_Pm, thb_act_out_Pd], axis=0) # (3,4,6)
                    
                    this_act_out = np.stack([hb_powers, hbo_powers, thb_powers], axis=0) # (3hb, 3px, 4band, 6ch)

                    act_out[act_num] = this_act_out # (3hb, 3px, 4band, 6ch)



            patients.append(act_out)
        
        patients = np.asarray(patients)
        print(patients.shape) # (26, 6, 3, 3, 4, 6)

        np.save('./inputs/full/' + levels[i] + '_fnirs_power_SMSD.npy', patients)
        print('>> npy saved.')
    ###################################################################################



def psd_feature_extraction():

    slice_sec_num = 10

    ###################################################################################
    # Sliced

    sliced_pth = './pickles/eeg-' + str(slice_sec_num) + 'sec_fnirs-full_baselinecorrected_fnirs_sliced.pkl'
    new_AD_sliced_pth = './pickles/new-AD_eeg-' + str(slice_sec_num) + 'sec_fnirs-full_baselinecorrected_fnirs_sliced.pkl'

    if not os.path.exists('./inputs/' + str(slice_sec_num) + '-sec/'):
        os.makedirs('./inputs/' + str(slice_sec_num) + '-sec/')

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

    # 1) generate EEG/fNIRS sliced input (Actually, EEG full + fNIRS sliced not used)
    EEG_RhythmicDecomposition_sliced(levels, all_levels_eeg, slice_sec=slice_sec_num)
    FNIRS_RhythmicDecomposition_sliced(levels, all_levels_fnirs, slice_sec=slice_sec_num)

    ###################################################################################
    # full
    full_pth = './pickles/eeg_full_fnirs_full_baselinecorrected.pkl'
    new_AD_full_pth = './pickles/new-AD_eeg_full_fnirs_full_baselinecorrected.pkl'

    if not os.path.exists('./inputs/full/'):
        os.makedirs('./inputs/full/')

    with open(full_pth, 'rb') as f:
        loaded_data = pickle.load(f) # ad, cn, ns, pre order
        print('>> ' + full_pth + ' loaded.')

    levels = ['AD', 'NORMAL', 'MCI']
    ad_eeg = loaded_data[0]['EEG']
    norm_eeg = loaded_data[1]['EEG']
    mci_eeg = loaded_data[3]['EEG']

    ad_fnirs = loaded_data[0]['fNIRs']
    norm_fnirs = loaded_data[1]['fNIRs']
    mci_fnirs = loaded_data[3]['fNIRs']


    with open(new_AD_full_pth, 'rb') as f:
        new_loaded_data = pickle.load(f) # ad, cn, ns, pre order
        print('>> ' + new_AD_full_pth + ' loaded.')    
    ad_eeg += new_loaded_data[0]['EEG']
    ad_fnirs += new_loaded_data[0]['fNIRs']

    print(len(ad_eeg), len(norm_eeg), len(mci_eeg))
    all_levels_eeg = [ad_eeg, norm_eeg, mci_eeg]
    print(len(ad_fnirs), len(norm_fnirs), len(mci_fnirs))
    all_levels_fnirs = [ad_fnirs, norm_fnirs, mci_fnirs]

    # 2) generate EEG/fNIRS full input (Actually, EEG full + fNIRS sliced not used)
    EEG_RhythmicDecomposition_full(levels, all_levels_eeg)
    FNIRS_RhythmicDecomposition_full(levels, all_levels_fnirs)


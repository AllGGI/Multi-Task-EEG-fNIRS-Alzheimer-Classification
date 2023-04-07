import pandas as pd
import numpy as np
import scipy
import pickle
import spkit as sp
import os


def EEG_RhythmicDecomposition_sliced(w_fnirs=False):
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
    '''
    ###################################################################################
    eeg_sampling_rate = 500
    slice_sec = 5
    sample_len = eeg_sampling_rate * slice_sec
    eeg_channels = 32

    for i, level_data in enumerate(all_levels_eeg): # ad 26 -> norm 64 -> mci 46
        patients = []
        for patient in level_data: # 26
            act_out = [[] for i in range(6)]
            for act_num, act_data in enumerate(patient): # 6 = RO, C1, C2, N1, N2, V
                if act_num == 0: # RO
                    data = act_data[:eeg_sampling_rate*60, :] # RO - 60 sec
                    # print(data.shape) # (30000, 32)
                    Px,Pm,Pd = sp.eeg.eeg_processing.RhythmicDecomposition(E=data, fs=eeg_sampling_rate)
                    # print(Px.shape, Pm.shape, Pd.shape) # (6 band, 32 ch) each
                    powers = np.stack([Px,Pm,Pd], axis=0)
                    # print(powers.shape)
                    
                    act_out[act_num] = powers # (3,6,32)

                else:
                    data_list = act_data
                    act_out_1_sample_Px, act_out_1_sample_Pm, act_out_1_sample_Pd = [], [], []
                    for data_sample in data_list: # 5 sec (2500, 32)
                        # print(data_sample.shape) # (2500, 32)
                        Px,Pm,Pd = sp.eeg.eeg_processing.RhythmicDecomposition(E=data_sample, fs=eeg_sampling_rate)
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

        np.save('./power_npy/sliced/' + levels[i] + '_eeg_power.npy', patients)
        print('>> npy saved.')
    ###################################################################################
    '''
    ###################################################################################
    fnirs_sampling_rate = 8
    slice_sec = 5
    sample_len = fnirs_sampling_rate * slice_sec
    fnirs_channels = 6
    fnirs_band = [[0.01, 0.04], [0.04, 0.15], [0.16, 0.6], [0.8, 1.5]] # vlfo, lfo, respi, heartbeat

    for i, level_data in enumerate(all_levels_fnirs): # ad 26 -> norm 64 -> mci 46
        patients = []
        for patient in level_data: # 26
            act_out = [[] for i in range(6)]
            for act_num, act_data in enumerate(patient): # 6 = RO, C1, C2, N1, N2, V
                if act_num == 0: # RO
                    hb_data = act_data['Hb'][:fnirs_sampling_rate*60, :] # RO - 60 sec (480, 6)
                    hbo_data = act_data['HbO'][:fnirs_sampling_rate*60, :] # RO - 60 sec (480, 6)
                    thb_data = act_data['THb'][:fnirs_sampling_rate*60, :] # RO - 60 sec (480, 6)

                    hb_Px,hb_Pm,hb_Pd = sp.eeg.eeg_processing.RhythmicDecomposition(E=hb_data, fs=fnirs_sampling_rate, fBands=fnirs_band)
                    hb_powers = np.stack([hb_Px,hb_Pm,hb_Pd], axis=0)

                    hbo_Px,hbo_Pm,hbo_Pd = sp.eeg.eeg_processing.RhythmicDecomposition(E=hbo_data, fs=fnirs_sampling_rate, fBands=fnirs_band)
                    hbo_powers = np.stack([hbo_Px,hbo_Pm,hbo_Pd], axis=0)

                    thb_Px,thb_Pm,thb_Pd = sp.eeg.eeg_processing.RhythmicDecomposition(E=thb_data, fs=fnirs_sampling_rate, fBands=fnirs_band)
                    thb_powers = np.stack([thb_Px,thb_Pm,thb_Pd], axis=0)

                    # print(hb_powers.shape) # (3,4,6)
                    powers = np.stack([hb_powers, hbo_powers, thb_powers], axis=0)
                    act_out[act_num] = powers # (3, 3,4,6)

                else:
                    hb_act_data = act_data['Hb']
                    hbo_act_data = act_data['HbO']
                    thb_act_data = act_data['THb']
                    # print(len(hb_act_data), len(hbo_act_data), len(thb_act_data)) # 25, 25, 25 / each (40, 6)
                    act_out_1_sample_Px, act_out_1_sample_Pm, act_out_1_sample_Pd = {'Hb':[], 'HbO':[], 'THb':[]}, {'Hb':[], 'HbO':[], 'THb':[]}, {'Hb':[], 'HbO':[], 'THb':[]}

                    for hb_data_sample, hbo_data_sample, thb_data_sample in zip(hb_act_data, hbo_act_data, thb_act_data): # dict
                        
                        hb_Px,hb_Pm,hb_Pd = sp.eeg.eeg_processing.RhythmicDecomposition(E=hb_data_sample, fs=fnirs_sampling_rate, fBands=fnirs_band, nperseg = 40)
                        hbo_Px,hbo_Pm,hbo_Pd = sp.eeg.eeg_processing.RhythmicDecomposition(E=hbo_data_sample, fs=fnirs_sampling_rate, fBands=fnirs_band, nperseg = 40)
                        thb_Px,thb_Pm,thb_Pd = sp.eeg.eeg_processing.RhythmicDecomposition(E=thb_data_sample, fs=fnirs_sampling_rate, fBands=fnirs_band, nperseg = 40)

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

        np.save('./power_npy/sliced/' + levels[i] + '_fnirs_power.npy', patients)
        print('>> npy saved.')
    ###################################################################################



def EEG_RhythmicDecomposition_full(w_fnirs=False):
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
    eeg_sampling_rate = 500
    slice_sec = 5
    sample_len = eeg_sampling_rate * slice_sec
    eeg_channels = 32

    for i, level_data in enumerate(all_levels): # ad 26 -> norm 64 -> mci 46
        patients = []
        for patient in level_data: # 26
            act_out = [[] for i in range(6)]
            for act_num, act_data in enumerate(patient): # 6 = RO, C1, C2, N1, N2, V
                if act_num == 0: # RO
                    data = act_data[:eeg_sampling_rate*60, :] # RO - 60 sec
                    # print(data.shape) # (30000, 32)
                    Px,Pm,Pd = sp.eeg.eeg_processing.RhythmicDecomposition(E=data, fs=eeg_sampling_rate)
                    # print(Px.shape, Pm.shape, Pd.shape) # (6 band, 32 ch) each
                    powers = np.stack([Px,Pm,Pd], axis=0)
                    # print(powers.shape)
                    
                    act_out[act_num] = powers # (3,6,32)

                else:
                    # print(act_data.shape) # (112112, 32)
                    Px,Pm,Pd = sp.eeg.eeg_processing.RhythmicDecomposition(E=act_data, fs=eeg_sampling_rate)

                    this_act_out = np.stack([Px, Pm, Pd], axis=0) # (3,6,32)
                    act_out[act_num] = this_act_out # (3,6,32)

            patients.append(act_out)
        
        patients = np.asarray(patients)
        print(i, patients.shape) # (26, 6, 3, 6, 32)

        np.save('./power_npy/full_sliced/' + levels[i] + '_eeg_power.npy', patients)
        print('>> npy saved.')


if __name__ == "__main__":

    full_sliced_pth = './pickles/eeg_fnirs_full_sliced_data.pkl'
    sliced_pth = './pickles/eeg_fnirs_sliced_data.pkl'
    sliced = True
    with_fnirs = True


    if sliced:

        if not os.path.exists('./power_npy/sliced/'):
            os.makedirs('./power_npy/sliced/')

        with open(sliced_pth, 'rb') as f:
            loaded_data = pickle.load(f) # ad, cn, ns, pre order
            print('>> eeg_fnirs_sliced_data.pkl loaded.')

        levels = ['AD', 'NORMAL', 'MCI']
        ad_eeg = loaded_data[0]['EEG']
        norm_eeg = loaded_data[1]['EEG']
        mci_eeg = loaded_data[3]['EEG']
        print(len(ad_eeg), len(norm_eeg), len(mci_eeg))
        all_levels_eeg = [ad_eeg, norm_eeg, mci_eeg]

        ad_fnirs = loaded_data[0]['fNIRs']
        norm_fnirs = loaded_data[1]['fNIRs']
        mci_fnirs = loaded_data[3]['fNIRs']
        print(len(ad_fnirs), len(norm_fnirs), len(mci_fnirs))
        all_levels_fnirs = [ad_fnirs, norm_fnirs, mci_fnirs]

        EEG_RhythmicDecomposition_sliced(w_fnirs=with_fnirs)

    else:

        if not os.path.exists('./power_npy/full_sliced/'):
            os.makedirs('./power_npy/full_sliced/')

        with open(full_sliced_pth, 'rb') as f:
            loaded_data = pickle.load(f) # ad, cn, ns, pre order
            print('>> eeg_fnirs_full_sliced_data.pkl loaded.')

        levels = ['AD', 'NORMAL', 'MCI']
        ad = loaded_data[0]['EEG']
        norm = loaded_data[1]['EEG']
        mci = loaded_data[3]['EEG']
        print(len(ad), len(norm), len(mci)) # 26, 64, 46
        all_levels = [ad, norm, mci]

        EEG_RhythmicDecomposition_full(w_fnirs=with_fnirs)
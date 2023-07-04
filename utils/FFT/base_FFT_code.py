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

# plot data
def plot_data(data, time, ch=6, title='fNIRs'):
    fig, axs = plt.subplots(ch, figsize=(10,11))
    fig.suptitle(title, fontsize=18)
    plt.subplots_adjust(left=0.1,
                        bottom=0.1, 
                        right=0.9, 
                        top=0.9, 
                        wspace=0.4, 
                        hspace=0.4)
    for i in range(ch):
        axs[i].plot(time[:100], data[:100,i])
        axs[i].set_title('ch: {}'.format(i), loc='right')

    plt.show()

# get_brain_wave_class function (fft for eeg)
def get_brain_wave_class(fs, data, plot=False):
    # Get real amplitudes of FFT (only in postive frequencies)
    fft_vals = np.absolute(np.fft.rfft(data))

    # Get frequencies for amplitudes in Hz
    fft_freq = np.fft.rfftfreq(len(data), 1.0/fs)

    # Define EEG bands
    eeg_bands = {'Delta': (1, 4),
                 'Theta': (4, 8),
                 'Alpha': (8, 13),
                 'Beta': (13, 30),
                 'Gamma': (30, 50)}

    # Take the mean of the fft amplitude for each EEG band
    eeg_band_fft = dict()
    for band in eeg_bands:  
        freq_ix = np.where((fft_freq >= eeg_bands[band][0]) & 
                           (fft_freq < eeg_bands[band][1]))[0]
        eeg_band_fft[band] = np.mean(fft_vals[freq_ix])

    # Plot the data (using pandas here cause it's easy)
    df = pd.DataFrame(columns=['band', 'val'])
    df['band'] = eeg_bands.keys()
    df['val'] = [eeg_band_fft[band] for band in eeg_bands]
    if plot:
        ax = df.plot.bar(x='band', y='val', legend=False)
        ax.set_xlabel("EEG band")
        ax.set_ylabel("Mean band Amplitude")
    return df

# fft function (for fnirs)
def fft(fs, data, plot=False):
    # Get real amplitudes of FFT (only in postive frequencies)
    fft_vals = np.absolute(np.fft.rfft(data))

    # Get frequencies for amplitudes in Hz
    fft_freq = np.fft.rfftfreq(len(data), 1.0/fs)
    
    # Define bands
    # freq_bands = [[0.2+i*0.2,0.4+i*0.2] for i in range(18)]
    fnirs_bands = {
        'VLFO':(0.01, 0.04),
        'LFO':(0.04, 0.15),
    }
    
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

def make_all_data_pkl(root_path, pths):
    # make pickle of all data
    data_list = [{},{},{},{}] # ad, cn, ns, pre order


    for i, path in enumerate(pths):
        patient_list = glob(os.path.join(root_path, path, '*'))
        # data_list[i]['EEG'] = [sorted(glob(os.path.join(x, 'EEG/*.mat'))) for x in patient_list]
        # data_list[i]['fNIRs'] = [sorted(glob(os.path.join(x, 'fNIRs/p_*.mat'))) for x in patient_list]
        data_list[i]['EEG'], data_list[i]['fNIRs'] = [], []
        eeg_act_dict = {'RO':0, 'C1':1, 'C2':2, 'N1':3, 'N2':4, 'V':5}
        fnirs_act_dict = {'resting':0, 'oddball1':1, 'oddball2':2, 'nback1':3, 'nback2':4, 'verbal':5}

        for x in patient_list:
            
            # EEG
            this_patient_pths = ['' for fc in range(6)]
            for file_name in glob(os.path.join(x, 'EEG/*.mat')):
                act_part = file_name.split('.')[-2][-3:]
                # print(file_name, act_part)
                if act_part[0] == '_': # _RO, _C1, _C2, _N1, _N2
                    this_patient_pths[eeg_act_dict[act_part[1:]]] = file_name
                else: # '_V'
                    this_patient_pths[eeg_act_dict['V']] = file_name

            data_list[i]['EEG'].append(this_patient_pths)


            # FNIRS (ex. p_01_resting.mat, 'resting' 'oddball1' 'oddball2' 'nback1' 'nback2' 'verbal')
            this_patient_pths = ['' for fc in range(6)]
            for file_name in glob(os.path.join(x, 'fNIRs/p_*.mat')):
                act_part = ((file_name.split('p_')[-1]).split('_')[-1]).split('.')[0]
                # print(file_name, act_part)
                this_patient_pths[fnirs_act_dict[act_part]] = file_name
            data_list[i]['fNIRs'].append(this_patient_pths)


        # print(len(data_list[i]['EEG']))
        # print(len(data_list[i]['fNIRs']))


    data = [{},{},{},{}] # ad, cn, ns, pre order
    for i, d_list in tqdm(enumerate(data_list)):
        eeg_file_list = d_list['EEG']
        dat = []
        for fn in eeg_file_list:
            if len(fn) != 6:
                continue
            # dict_keys: ['__header__', '__version__', '__globals__', 'data']
            dat.append([io.loadmat(f)['data'] for f in fn])
        
        fnirs_file_list = d_list['fNIRs']
        mat = []
        for fn in fnirs_file_list:
            if len(fn) != 6:
                continue
            # dict_keys: ['__header__', '__version__', '__globals__', 'data850', 'data735' or 'data730', 'time',
            #             'Target', 'Nontarget', 't', 'Response', 'HbO', 'Hb', 'THb']
            # 2nd_year: 'data730', 3rd_year: 'data735'

            # mat.append([io.loadmat(f) for f in fn]) # cannot run this because of various key names
            # Renaming some keys (unnecessary now)
            renamed_lst = []
            for f in fn:
                # print(f)
                loaded_mat = io.loadmat(f)
                if 'data730' in loaded_mat.keys():
                    loaded_mat['data730or5'] = loaded_mat.pop('data730')
                elif 'data735' in loaded_mat.keys():
                    loaded_mat['data730or5'] = loaded_mat.pop('data735')
                elif 'int_730' in loaded_mat.keys():
                    loaded_mat['data730or5'] = loaded_mat.pop('int_730')
                
                if 'int_850' in loaded_mat.keys():
                    loaded_mat['data850'] = loaded_mat.pop('int_850')

                if 'nirs_time' in loaded_mat.keys():
                    loaded_mat['time'] = loaded_mat.pop('nirs_time')

                renamed_lst.append(loaded_mat)

            mat.append(renamed_lst)
        
        data[i]['EEG'] = dat
        data[i]['fNIRs'] = mat

    # make save folder
    if not os.path.exists('./pickles'):
        os.makedirs('./pickles')

    # save all data pickle
    with open('./pickles/all_data.pkl', 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        print('>> 1-1) all_data.pkl saved.')

    # load all data pickle
    with open('./pickles/all_data.pkl', 'rb') as f:
        loaded_data = pickle.load(f) # ad, cn, ns, pre order
        print('>> 1-1) all_data.pkl loaded.')

    return loaded_data


# # Use EEG ('data'), fNIRs('time', 'Hb', 'HbO', 'THb')
# def make_essential_data_pkl(pths):
#     # make pickle of essential data
#     data_list = [{},{},{},{}] # ad, cn, ns, pre order

#     for i, path in enumerate(pths):
#         patient_list = glob(os.path.join(root_path, path, '*'))
#         data_list[i]['EEG'] = [sorted(glob(os.path.join(x, 'EEG/*.mat'))) for x in patient_list]
#         data_list[i]['fNIRs'] = [sorted(glob(os.path.join(x, 'fNIRs/p_*.mat'))) for x in patient_list]

#     data = [{},{},{},{}] # ad, cn, ns, pre order
#     for i, d_list in tqdm(enumerate(data_list)):
#         eeg_file_list = d_list['EEG']
#         dat = []
#         for fn in eeg_file_list:
#             if len(fn) != 6:
#                 continue
#             # dict_keys: ['__header__', '__version__', '__globals__', 'data']
#             dat.append([io.loadmat(f)['data'] for f in fn])
        
#         fnirs_file_list = d_list['fNIRs']
#         mat = []
#         for fn in fnirs_file_list:
#             if len(fn) != 6:
#                 continue
#             # dict_keys: ['__header__', '__version__', '__globals__', 'data850', 'data735' or 'data730', 'time',
#             #             'Target', 'Nontarget', 't', 'Response', 'HbO', 'Hb', 'THb']
#             # 2nd_year: 'data730', 3rd_year: 'data735'

#             # mat.append([io.loadmat(f) for f in fn]) # cannot run this because of various key names
#             # Renaming some keys
#             renamed_lst = []
#             for f in fn:
#                 loaded_mat = io.loadmat(f)
#                 if 'nirs_time' in loaded_mat.keys():
#                     loaded_mat['time'] = loaded_mat.pop('nirs_time')

#                 essential_keys = ['time', 'Hb', 'HbO', 'THb']
#                 copied_mat = copy.deepcopy(loaded_mat) # prevent dictionary changed size during iteration
#                 for this_key in loaded_mat.keys():
#                     if this_key not in essential_keys: del(copied_mat[this_key])

#                 renamed_lst.append(copied_mat)

#             mat.append(renamed_lst)
        
#         data[i]['EEG'] = dat
#         data[i]['fNIRs'] = mat

#     # make save folder
#     if not os.path.exists('./pickles'):
#         os.makedirs('./pickles')

#     # save essential data pickle
#     with open('./pickles/essential_data.pkl', 'wb') as f:
#         pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
#         print('>> 1-2) essential_data.pkl saved.')

#     # load all data pickle
#     with open('./pickles/essential_data.pkl', 'rb') as f:
#         loaded_data = pickle.load(f) # ad, cn, ns, pre order
#         print('>> 1-2) essential_data.pkl loaded.')

#     return loaded_data

# Use EEG ('data'), fNIRs('time', 'Hb', 'HbO', 'THb')
def fft_all_data(data):

    patient_nums = [len(data[i]['fNIRs']) for i in range(4)]
    eeg_ch = 32
    fnirs_ch = 6 # 4 long + 2 short
    patient_all_data = []
    for label in range(4):
        for patient_num in range(patient_nums[label]):
            act_all_data = []
            for act_num in range(6):
                # 1. time = [-1] or 2. time = [-1] - [0]
                # 1.
                # time = data[label]['fNIRs'][patient_num][act_num]['time'][0][-1] 
                # 2.
                start_time = data[label]['fNIRs'][patient_num][act_num]['time'][0][0]
                end_time = data[label]['fNIRs'][patient_num][act_num]['time'][0][-1]
                time = end_time - start_time

                # print(label, patient_num, act_num, time)
                # fft EEG for 32 channel
                eeg_allch = []
                for ch in range(eeg_ch):
                    eeg = np.array(data[label]['EEG'][patient_num][act_num])[:,ch] # (x, 32)
                    # print(len(eeg)/time) # see sampling rate
                    eeg_1ch_df = get_brain_wave_class(len(eeg)/time, eeg) # param: sampling rate = 500 (init), eeg data
                    eeg_allch.append([eeg_1ch_df['val'][i] for i in range(len(eeg_1ch_df['val']))])

                fnirs_allch = []
                # fft fNIRS for 6 channel
                for ch in range(fnirs_ch):
                    fnirs = data[label]['fNIRs'][patient_num][act_num] # (x, 6)
                    hb = np.array(fnirs['Hb'])[:,ch]
                    hbo = np.array(fnirs['HbO'])[:,ch]
                    thb = np.array(fnirs['THb'])[:,ch]
                    
                    hb_fft = fft(len(hb)/time, hb)
                    hbo_fft = fft(len(hbo)/time, hbo)
                    thb_fft = fft(len(thb)/time, thb)
                    fnirs_allch.append([hb_fft, hbo_fft, thb_fft])
                act_all_data.append([eeg_allch, fnirs_allch])
            patient_all_data.append([label, act_all_data])

    # save
    with open('./pickles/fft_data.pkl', 'wb') as f:
        pickle.dump(patient_all_data, f, pickle.HIGHEST_PROTOCOL)

    # load
    with open('./pickles/fft_data.pkl', 'rb') as f:
        patient_all_data = pickle.load(f) # ad, cn, ns, pre order

    # # !!! POSTPROCESSING !!! : Flatten 'fft_data' for dataframe
    # # patient_all_data[{patient_num}][{0:label, 1:data}][{act_num}][{0:eeg,1:fnirs}] 
    # # eeg 32ch, 5 properties, fnirs 6x3ch, 18 properties
    # flattened_data = []
    # for i in range(len(patient_all_data)):
    #     p_f_data = []
    #     label = patient_all_data[i][0]
    #     one_patient_data = patient_all_data[i][1]
    #     all_eeg_data = []
    #     all_fnirs_data = []
    #     for act_num in range(6):
    #         act_data = one_patient_data[act_num]
    #         eeg_data = act_data[0]
    #         fnirs_data = act_data[1]
    #         all_eeg_data += [eeg_data[j][k] for j in range(len(eeg_data)) for k in range(5)]
    #         all_fnirs_data += [fnirs_data[j][k][w] for j in range(len(fnirs_data)) for k in range(3) for w in range(4)]

    #     flattened_data.append([label] + all_eeg_data + all_fnirs_data)

    return patient_all_data # not flattened


def fft_all_data_norm(data):

    patient_nums = [len(data[i]['fNIRs']) for i in range(4)]

    # print(patient_nums)

    eeg_ch = 32
    fnirs_ch = 6
    patient_all_data_normalized = []
    for label in range(4):
        for patient_num in range(patient_nums[label]):
            act_all_data = []
            for act_num in range(6):
                # 1. time = [-1] or 2. time = [-1] - [0]
                # 1.
                # time = data[label]['fNIRs'][patient_num][act_num]['time'][0][-1] 
                # 2.
                start_time = data[label]['fNIRs'][patient_num][act_num]['time'][0][0]
                end_time = data[label]['fNIRs'][patient_num][act_num]['time'][0][-1]
                time = end_time - start_time

                # print(label, patient_num, act_num, time)
                # fft EEG for 32 channel
                eeg_allch = []
                for ch in range(eeg_ch):
                    # print(label, patient_num, act_num, np.array(data[label]['EEG'][patient_num][act_num]).shape)
                    eeg = np.array(data[label]['EEG'][patient_num][act_num])[:,ch] # (x, 32)

                    # print(eeg.shape) # (197865,) ~
                    # print(len(eeg)/time) # 490~500

                    eeg_1ch_df = get_brain_wave_class(len(eeg)/time, eeg)
                    eeg_1ch_df = [eeg_1ch_df['val'][i] for i in range(len(eeg_1ch_df['val']))]
                    
                    # normalize
                    norm = np.linalg.norm(eeg_1ch_df)
                    eeg_1ch_df = eeg_1ch_df/norm
                    
                    eeg_allch.append(eeg_1ch_df.tolist())
                    
                fnirs_allch = []
                # fft fNIRS for 6 channel
                for ch in range(fnirs_ch):
                    fnirs = data[label]['fNIRs'][patient_num][act_num] # (x, 6)
                    hb = np.array(fnirs['Hb'])[:,ch]
                    hbo = np.array(fnirs['HbO'])[:,ch]
                    thb = np.array(fnirs['THb'])[:,ch]
                    
                    hb_fft = fft(len(hb)/time, hb)
                    hbo_fft = fft(len(hbo)/time, hbo)
                    thb_fft = fft(len(thb)/time, thb)
                    
                    # normalize
                    norm = np.linalg.norm(hb_fft)
                    hb_fft = hb_fft/norm
                    norm = np.linalg.norm(hbo_fft)
                    hbo_fft = hbo_fft/norm
                    norm = np.linalg.norm(thb_fft)
                    thb_fft = thb_fft/norm
                    
                    fnirs_allch.append([hb_fft.tolist(), hbo_fft.tolist(), thb_fft.tolist()])
                act_all_data.append([eeg_allch, fnirs_allch])
            patient_all_data_normalized.append([label, act_all_data])

    # save
    with open('./pickles/fft_data_norm.pkl', 'wb') as f:
        pickle.dump(patient_all_data_normalized, f, pickle.HIGHEST_PROTOCOL)

    # load
    with open('./pickles/fft_data_norm.pkl', 'rb') as f:
        patient_all_data_normalized = pickle.load(f) # ad, cn, ns, pre order

    # !!! POSTPROCESSING !!! : Flatten 'fft_data_norm' for dataframe
    # patient_all_data[{patient_num}][{0:label, 1:data}][{act_num}][{0:eeg,1:fnirs}] 
    # eeg 32ch, 5 properties, fnirs 6x3ch, 18 properties
    # FINAL FLATTENED FORMAT = label + eeg data 32 ch * 5 band *  + fnirs data 6 * 3 (hb, hbo, thb)ch * 4 band
    flattened_norm_data = []
    for i in range(len(patient_all_data_normalized)):
        p_f_data = []
        label = patient_all_data_normalized[i][0]
        one_patient_data = patient_all_data_normalized[i][1]
        all_eeg_data = []
        all_fnirs_data = []
        for act_num in range(6):
            act_data = one_patient_data[act_num]
            eeg_data = act_data[0]
            fnirs_data = act_data[1]
            all_eeg_data += [eeg_data[j][k] for j in range(len(eeg_data)) for k in range(5)] # EEG 5 bands, j(channel) k(band)
            all_fnirs_data += [fnirs_data[j][k][w] for j in range(len(fnirs_data)) for k in range(3) for w in range(4)] # fnirs 4 bands

        flattened_norm_data.append([label] + all_eeg_data + all_fnirs_data)

    return flattened_norm_data # flattened 'patient_all_data_normalized'


if __name__ == "__main__":

    ####################################################################################################
    # 1) Make all data -> pickle

    # make data list
    root_path = "D:\치매감지\EEG_fNIRs_dataset\Sorted_Dataset\Sorted_alzh_dataset".replace('\\', '/')
    # print(root_path)
    ad_path = "AD" # 치매
    cn_path = "NORMAL" # 정상
    ns_path = "NP" # 무증상 = ns
    pre_path = "MCI" # 전조증상 = pre
    all_path = [ad_path, cn_path, ns_path, pre_path]

    # 1-1) Make 'all_data.pkl'
    if not os.path.isfile('./pickles/all_data.pkl'):
        all_data = make_all_data_pkl(root_path, all_path) # save & load all data pickle
    else:
        with open('./pickles/all_data.pkl', 'rb') as f:
            all_data = pickle.load(f) # ad, cn, ns, pre order
            print('>> 1-1) Existing all_data.pkl loaded.')

    # # 1-2) Make 'essential_data.pkl' (fnirs: dict_keys(['time', 'HbO', 'Hb', 'THb']))
    # if not os.path.isfile('./pickles/essential_data.pkl'):
    #     all_data = make_essential_data_pkl(root_path, all_path) # save & load all data pickle
    # else:
    #     with open('./pickles/essential_data.pkl', 'rb') as f:
    #         all_data = pickle.load(f) # ad, cn, ns, pre order
    #         print('>> 1-2) Existing essential_data.pkl loaded.')

    ####################################################################################################
    # dimension

    # EEG
    ### all_data[{0:ad,1:cn,2:ns,3:pre}]['EEG'][{patient_num}][{act_num}]
    # print(all_data[0]['EEG'][0][0].shape) # (31520, 32)
    # print(all_data[0]['EEG'][0][1].shape) # (142112, 32)
    # print(all_data[0]['EEG'][0][2].shape) # (141208, 32)
    # print(all_data[0]['EEG'][0][3].shape) # (130008, 32)
    # print(all_data[0]['EEG'][0][4].shape) # (130752, 32)
    # print(all_data[0]['EEG'][0][5].shape) # (197928, 32)

    # fNIRs
    ### all_data[{0:ad,1:cn,2:ns,3:pre}]['fNIRs'][{patient_num}][{act_num}][{'data730','data850','Hb','HbO','THb','time'}]
    ### 6ch 데이터: 첫 3개 오른쪽 전두엽, 다음 3개 왼쪽 전두엽
    ### 'data730', 'data850': 730,850nm 파장 빛 측정 세기, 0~4095
    ### HbO: 산화헤모글로빈, Hb: 환원헤모글로빈, THb: 총 헤모글로빈 # 상대적인 농도 변화
    ### Sampling rate 8Hz
    # print(all_data[0]['fNIRs'][0][0]['data730'].shape) # (587, 6)
    # print(all_data[0]['fNIRs'][0][1]['data730'].shape) # (2299, 6)
    # print(all_data[0]['fNIRs'][0][2]['data730'].shape) # (2310, 6)
    # print(all_data[0]['fNIRs'][0][3]['data730'].shape) # (2142, 6)
    # print(all_data[0]['fNIRs'][0][4]['data730'].shape) # (2147, 6)
    # print(all_data[0]['fNIRs'][0][5]['data730'].shape) # (3206, 6)
    # print(all_data[0]['fNIRs'][0][0]['Hb'].shape) # (587, 6)
    # print(all_data[0]['fNIRs'][0][0]['HbO'].shape) # (587, 6)
    # print(all_data[0]['fNIRs'][0][0]['THb'].shape) # (587, 6)

    # print(all_data[0]['fNIRs'][0][0]['Hb'][:100, 0].shape) # (100,)

    ####################################################################################################
    # 2) Plot fNIRs & EEG 1st data

    # # fNIRs plot
    # plot_data(all_data[0]['fNIRs'][0][0]['data730or5'], data[0]['fNIRs'][0][0]['time'][0], 6, 'fNIRs, 730 or 735nm')
    # plot_data(all_data[0]['fNIRs'][0][0]['data850'], data[0]['fNIRs'][0][0]['time'][0], 6, 'fNIRs, 850nm')
    # plot_data(all_data[0]['fNIRs'][0][0]['Hb'], data[0]['fNIRs'][0][0]['time'][0], 6, 'fNIRs, Hb')
    # plot_data(all_data[0]['fNIRs'][0][0]['HbO'], data[0]['fNIRs'][0][0]['time'][0], 6, 'fNIRs, HbO')
    # plot_data(all_data[0]['fNIRs'][0][0]['THb'], data[0]['fNIRs'][0][0]['time'][0], 6, 'fNIRs, THb')

    # # EEG plot
    # plot_data(all_data[0]['EEG'][0][0], np.linspace(0, 1, 100), 6, 'EEG')

    ####################################################################################################
    # # 3) FFT all data
    # patient_all_data = fft_all_data(all_data)
    # print('>> 3) FFT all data finished.')

    ####################################################################################################
    # 4) FFT all data + norm
    flattened_norm_data = fft_all_data_norm(all_data)
    print('>> 4) FFT all data + norm finished.')

    ####################################################################################################
    # 5) making column
    column = ['label']
    eeg_band_names = ['delta','theta','alpha','beta','gamma']
    fnirs_band_names = ['vlfo', 'lfo', 'respi', 'heartb']
    fnirs_props = ['Hb', 'HbO','THb']
    for act_num in range(6):
        for ch in range(32):
            for band in range(len(eeg_band_names)): # 5 bands
                column += ['eeg_act-{}_ch-{}_band-{}'.format(act_num, ch, eeg_band_names[band])]
    for act_num in range(6):
        for ch in range(6):
            for prop in fnirs_props:
                for band in range(len(fnirs_band_names)): # 4 bands
                    column += ['fnirs_act-{}_ch-{}_prop-{}_band-{}'.format(act_num, ch, prop, fnirs_band_names[band])]

    ####################################################################################################
    # 6) save csv
    # make save folder
    if not os.path.exists('./csv'):
        os.makedirs('./csv')
    df = pd.DataFrame(flattened_norm_data, columns=column)
    df.to_csv('./csv/fft_data_norm.csv', sep=',')
    print('>> 6) csv saved.')


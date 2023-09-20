
#####################################################################################
################## Slice 5 sec samples from each target idx #########################
#####################################################################################

import os
import scipy
from scipy import io
import numpy as np
from glob import glob
from tqdm import tqdm
import pickle

global_slice_sec = 10

eeg_sampling_rate = 500
fnirs_sampling_rate = 8
# targ_idx_choice_num = 5 # num to choice targ_idx

# No trigger file: [AD s107, NORMAL s097, NORMAL s099, NORMAL s100]
# No verbal 'target' trigger: [NORMAL s003, NORMAL s008, NORMAL s010, NP s005, NP s009, MCI s007]
# Act len != 6 in FNIRS: [MCI_s134]
eeg_passed_patients = []

targets = {'C1':[], 'C2':[], 'N1':[], 'N2':[]}
avg_target_terms = {'C1':[], 'C2':[], 'N1':[], 'N2':[]}


# if trigger key == 'eeg'
def trigger_named_eeg(act_name, trig_file, eeg_mat, fnirs_mat):

    '''
        return: sliced_data, FLAG
            1. sliced_data = sliced EEG data (empty if error occur)
            2. FLAG = True if well sliced, else False
    '''

    act_dict = {'RO':'resting', 'C1':'oddball1', 'C2':'oddball2', 'N1':'nback1', 'N2':'nback2', 'V':'verbal'}
    trig = trig_file['eeg'][0]
    act_trigger = trig[act_dict[act_name]] # ex) 'oddball1' data

    targ_nontarg = []
    targ_idx, non_targ_idx = [], []
    if act_name == 'C1' or act_name == 'C2':
        targ = act_trigger[0]['event'][0][0][0]['target'][0][0]
        # print(targ.shape) # (141160,)
        targ_idx = np.nonzero(targ)[0]
        # print(act_name, targ_idx)

        ####################################################################
        # For Checking mean target idx location
        # a = [targ_idx[i+1]-targ_idx[i] for i in range(len(targ_idx)-1)]
        # targets[act_name].append(len(targ_idx))
        # avg_target_terms[act_name].append(np.mean(a))
        ####################################################################

        non_targ = act_trigger[0]['event'][0][0][0]['nontarget'][0][0]
        # print(act_name, non_targ.shape) # (141160,)
        non_targ_idx = np.nonzero(non_targ)[0]
        # print(non_targ_idx[0])

        targ_nontarg = targ + non_targ

    elif act_name == 'N1' or act_name == 'N2':
        targ = act_trigger[0]['event'][0][0][0]['target'][0].squeeze()
        # print(act_name, targ.shape) # (130544,)
        targ_idx = np.nonzero(targ)[0]
        # print(act_name, targ_idx)

        ####################################################################
        # For Checking mean target idx location
        # a = [targ_idx[i+1]-targ_idx[i] for i in range(len(targ_idx)-1)]
        # targets[act_name].append(len(targ_idx))
        # avg_target_terms[act_name].append(np.mean(a))
        ####################################################################

        # print('>> nontarget:')
        non_targ = act_trigger[0]['event'][0][0][0]['nontarget'][0].squeeze()
        # print(act_name, non_targ.shape) # (144300,)
        non_targ_idx = np.nonzero(non_targ)[0]
        # print(non_targ_idx[0])

        targ_nontarg = targ + non_targ

    elif act_name == 'V':
        try:
            targ = act_trigger[0]['event'][0][0][0]['target'][0][0]
            # print(act_name, targ.shape) # (144300,)
            targ_idx = np.nonzero(targ)[0]
            targ_nontarg = targ # nontarg == [Verbal has no nontarget]
            # print(act_name, targ_idx)

        except:
            # print(act_trigger[0]['event'][0][0].dtype) # [('storageTime', 'O')], no 'target'
            return np.empty([1,1]), {}, False

    targ_nontarg_idx = np.nonzero(targ_nontarg)[0]
    # print(targ_nontarg_idx) # start == 15000

    assert targ_nontarg.shape[0] == eeg_mat['data'].shape[0]

    #########################################################################
    # TODO: change here if you don't want to start at 30 sec (=15000 or 14985)
    # Slice EEG/fNIRs data

    '''
        Option 3)
        C1 - from all targets, slice 5 sec (2500)
        C2 - from all targets, slice 5 sec (2500)
        N1 - from all targets, slice 5 sec (2500)
        N2 - from all targets, slice 5 sec (2500)
        V - from all targets, slice 30 sec (15000)
    '''
    eeg_initial_start_idx = targ_nontarg_idx[0] # 15000
    sliced_eeg_data, sliced_fnirs_data = [], {'Hb':[], 'HbO':[], 'THb':[]}

    ###############################################################
    # EEG Baseline correction (- mean of initial 30 sec)
    eeg_data = eeg_mat['data'] - np.mean(eeg_mat['data'][:eeg_initial_start_idx,:], axis=0) # (32,)

    # FNIRS Baseline correction (- initial value)
    fnirs_initial_start_idx = int((eeg_initial_start_idx / 500) * 8) # ex) 15000/500 * 8 = 240
    hb_fnirs_data = fnirs_mat['Hb'] - np.mean(fnirs_mat['Hb'][:fnirs_initial_start_idx,:], axis=0) # (6,)
    hbo_fnirs_data = fnirs_mat['HbO'] - np.mean(fnirs_mat['HbO'][:fnirs_initial_start_idx,:], axis=0)
    thb_fnirs_data = fnirs_mat['THb'] - np.mean(fnirs_mat['THb'][:fnirs_initial_start_idx,:], axis=0)
    ###############################################################

    for idx in range(len(targ_idx)):
        eeg_start_idx = targ_idx[idx]
        fnirs_start_idx = int((eeg_start_idx / 500) * 8) # ex) (16000-15000)/500 *8 = 16

        if act_name == 'V': # substract just before 30 sec mean
            # eeg
            eeg_sample = eeg_data[eeg_start_idx:eeg_start_idx+15000, :] # (15000,32)

            # fnirs
            hb_sample = hb_fnirs_data[fnirs_start_idx:fnirs_start_idx+240, :] # (240=30sec, 6)
            hbo_sample = hbo_fnirs_data[fnirs_start_idx:fnirs_start_idx+240, :] # (240=30sec, 6)
            thb_sample = thb_fnirs_data[fnirs_start_idx:fnirs_start_idx+240, :] # (240=30sec, 6)

        else:
            # eeg
            eeg_sample = eeg_data[eeg_start_idx:eeg_start_idx+(eeg_sampling_rate*global_slice_sec), :] # (1500, 32)

            # fnirs (+24: after delay 3 sec)
            hb_sample = hb_fnirs_data[fnirs_start_idx:fnirs_start_idx+24+(fnirs_sampling_rate*global_slice_sec), :] # (10 sec, 6)
            hbo_sample = hbo_fnirs_data[fnirs_start_idx:fnirs_start_idx+24+(fnirs_sampling_rate*global_slice_sec), :] # (10 sec, 6)
            thb_sample = thb_fnirs_data[fnirs_start_idx:fnirs_start_idx+24+(fnirs_sampling_rate*global_slice_sec), :] # (10 sec, 6)


        sliced_eeg_data.append(eeg_sample)

        # Option 1) FNIRS sliced data
        sliced_fnirs_data['Hb'].append(hb_sample)
        sliced_fnirs_data['HbO'].append(hbo_sample)
        sliced_fnirs_data['THb'].append(thb_sample)


    # Option 2) FNIRS full data
    # sliced_fnirs_data['Hb'] = hb_fnirs_data
    # sliced_fnirs_data['HbO'] = hbo_fnirs_data
    # sliced_fnirs_data['THb'] = thb_fnirs_data


    return sliced_eeg_data, sliced_fnirs_data, True # list, dict of lists, bool



# if trigger key == 'trigger'
def trigger_named_trigger(act_name, trig_file, eeg_mat, fnirs_mat):
    
    '''
        return: sliced_data, FLAG
            1. sliced_data = sliced EEG data (empty if error occur)
            2. FLAG = True if well sliced, else False
    '''
    act_dict = {'RO':'resting', 'C1':'oddball1', 'C2':'oddball2', 'N1':'nback1', 'N2':'nback2', 'V':'verbal'}
    trig = trig_file['trigger'][0]
    # print(trig.dtype)
    act_trigger = trig[act_dict[act_name]] # ex) 'oddball1' data

    targ_nontarg = []
    targ_idx, non_targ_idx = [], []
    if act_name == 'C1' or act_name == 'C2':
        targ = act_trigger[0]['target'][0][0][0]
        # print(act_name, targ.shape) # (144300,)
        targ_idx = np.nonzero(targ)[0]

        ####################################################################
        # For Checking mean target idx location
        # a = [targ_idx[i+1]-targ_idx[i] for i in range(len(targ_idx)-1)]
        # targets[act_name].append(len(targ_idx))
        # avg_target_terms[act_name].append(np.mean(a))
        ####################################################################


        # print('>> nontarget:')
        non_targ = act_trigger[0]['nontarget'][0][0][0]
        # print(non_targ.shape) # (144300,)
        non_targ_idx = np.nonzero(non_targ)[0]
        # print(non_targ_idx[0])

        targ_nontarg = targ + non_targ
    
    elif act_name == 'N1' or act_name == 'N2':
        targ = act_trigger[0]['target'][0][0].squeeze()
        # print(act_name, targ.shape) # (144300,)
        targ_idx = np.nonzero(targ)[0]

        ####################################################################
        # For Checking mean target idx location
        # a = [targ_idx[i+1]-targ_idx[i] for i in range(len(targ_idx)-1)]
        # targets[act_name].append(len(targ_idx))
        # avg_target_terms[act_name].append(np.mean(a))
        ####################################################################


        # print('>> nontarget:')
        non_targ = act_trigger[0]['nontarget'][0][0].squeeze()
        # print(act_name, non_targ.shape) # (144300,)
        non_targ_idx = np.nonzero(non_targ)[0]
        # print(non_targ_idx[0])

        targ_nontarg = targ + non_targ

    elif act_name == 'V':
        try:
            targ = act_trigger[0]['target'][0][0][0]
            # print(act_name, targ.shape) # (144300,)
            targ_idx = np.nonzero(targ)[0]
            targ_nontarg = targ # nontarg == [Verbal has no nontarget]
            # print(act_name, targ_idx)

        except:
            # print(act_trigger[0]['event'][0][0].dtype) # [('storageTime', 'O')], no 'target'
            return np.empty([1,1]), {}, False


    targ_nontarg_idx = np.nonzero(targ_nontarg)[0]
    # print(targ_nontarg_idx) # start == 15000


    #########################################################################
    # Slice EEG data  
    '''
        Option 3)
        C1 - from all targets, slice 5 sec (2500)
        C2 - from all targets, slice 5 sec (2500)
        N1 - from all targets, slice 5 sec (2500)
        N2 - from all targets, slice 5 sec (2500)
    '''
    eeg_initial_start_idx = targ_nontarg_idx[0] # 15000
    sliced_eeg_data, sliced_fnirs_data = [], {'Hb':[], 'HbO':[], 'THb':[]}

    ###############################################################
    # EEG Baseline correction (- mean of initial 30 sec)
    eeg_data = eeg_mat['data'] - np.mean(eeg_mat['data'][:eeg_initial_start_idx,:], axis=0) # (32,)

    # FNIRS Baseline correction (- initial value)
    fnirs_initial_start_idx = int((eeg_initial_start_idx / 500) * 8) # ex) 15000/500 * 8 = 240
    hb_fnirs_data = fnirs_mat['Hb'] - np.mean(fnirs_mat['Hb'][:fnirs_initial_start_idx,:], axis=0) # (6,)
    hbo_fnirs_data = fnirs_mat['HbO'] - np.mean(fnirs_mat['HbO'][:fnirs_initial_start_idx,:], axis=0)
    thb_fnirs_data = fnirs_mat['THb'] - np.mean(fnirs_mat['THb'][:fnirs_initial_start_idx,:], axis=0)
    ###############################################################

    for idx in range(len(targ_idx)):
        eeg_start_idx = targ_idx[idx]
        fnirs_start_idx = int((eeg_start_idx / 500) * 8) # ex) (16000-15000)/500 *8 = 16

        if act_name == 'V': # substract just before 30 sec mean
            # eeg
            eeg_sample = eeg_data[eeg_start_idx:eeg_start_idx+15000, :] # (15000,32)

            # fnirs
            hb_sample = hb_fnirs_data[fnirs_start_idx:fnirs_start_idx+240, :] # (240=30sec, 6)
            hbo_sample = hbo_fnirs_data[fnirs_start_idx:fnirs_start_idx+240, :] # (240=30sec, 6)
            thb_sample = thb_fnirs_data[fnirs_start_idx:fnirs_start_idx+240, :] # (240=30sec, 6)

        else:
            # eeg
            eeg_sample = eeg_data[eeg_start_idx:eeg_start_idx+(eeg_sampling_rate*global_slice_sec), :] # (1500, 32)

            # fnirs (+24: after delay 3 sec)
            hb_sample = hb_fnirs_data[fnirs_start_idx:fnirs_start_idx+24+(fnirs_sampling_rate*global_slice_sec), :] # (10 sec, 6)
            hbo_sample = hbo_fnirs_data[fnirs_start_idx:fnirs_start_idx+24+(fnirs_sampling_rate*global_slice_sec), :] # (10 sec, 6)
            thb_sample = thb_fnirs_data[fnirs_start_idx:fnirs_start_idx+24+(fnirs_sampling_rate*global_slice_sec), :] # (10 sec, 6)


        sliced_eeg_data.append(eeg_sample)

        # Option 1) FNIRS sliced data
        sliced_fnirs_data['Hb'].append(hb_sample)
        sliced_fnirs_data['HbO'].append(hbo_sample)
        sliced_fnirs_data['THb'].append(thb_sample)


    # Option 2) FNIRS full data
    # sliced_fnirs_data['Hb'] = hb_fnirs_data
    # sliced_fnirs_data['HbO'] = hbo_fnirs_data
    # sliced_fnirs_data['THb'] = thb_fnirs_data


    return sliced_eeg_data, sliced_fnirs_data, True # list, dict of lists, bool

    

def make_sliced_data_pkl(levels, root_path, trig_list):
    # make pickle of all data
    data_list = [{},{},{},{}] # ad, cn, ns, pre order

    for i, path in enumerate(levels):
        patient_list = glob(os.path.join(root_path, path, '*'))
        data_list[i]['EEG'] = [sorted(glob(os.path.join(x, 'EEG/*.mat'))) for x in patient_list]
        data_list[i]['fNIRs'] = [sorted(glob(os.path.join(x, 'fNIRs/p_*.mat'))) for x in patient_list]

    data = [{},{},{},{}] # ad, cn, ns, pre order
    act_save_order = {'RO':0, 'C1':1, 'C2':2, 'N1':3, 'N2':4, 'V':5}
    eeg_fnirs_act_matching = {'RO':'p_01_resting', 'C1':'p_02_oddball1', 'C2':'p_03_oddball2', 'N1':'p_04_nback1', 'N2':'p_05_nback2', 'V':'p_06_verbal'}

    for i, d_list in tqdm(enumerate(data_list)): # AD -> Normal -> NP -> MCI order

        ################################### EEG ##########################################
        eeg_file_list = d_list['EEG']
        dat, mat = [], []
        for fn in eeg_file_list: # Patients
            # dict_keys: ['__header__', '__version__', '__globals__', 'data']
            patient_name = fn[0].replace('\\', '/').split('/')[-3]
            level = fn[0].replace('\\', '/').split('/')[-4]
            if len(fn) != 6:
                print('\n>> EEG Pass: Act len != 6 ->', level, patient_name)
                eeg_passed_patients.append(level + '_' + patient_name)
                continue
            ####################################################################
            # !! HARD CODING HERE !! Pass 'MCI_s134' since this not has 6 act in FNIRS
            if level == 'MCI' and patient_name == 's134':
                print('\n>> EEG Pass: (FNIRS) Act len != 6 ->', level, patient_name)
                eeg_passed_patients.append(level + '_' + patient_name)
                continue
            ####################################################################
            # print(level, patient_name)
            
            ####################################################################
            # Find index of 'this patient's trigger file' from trigger list
            lst = trig_list[level]
            trig_matched_idx = -1 # index of 'this patient's trigger file'
            for idx, pth in enumerate(lst):
                if (patient_name.upper() in pth) or (patient_name in pth): # either 's006' or 'S006'
                    trig_matched_idx = idx
                    break
            
            if trig_matched_idx == -1: # pass this patient (since no trigger file) => 'normal 97, 99, 100'
                print('\n>> EEG Pass: no trigger file ->', level, patient_name)
                eeg_passed_patients.append(level + '_' + patient_name)
                # print('*' * 100)
                continue
            
            ####################################################################            
            this_patient_trig_file = io.loadmat(lst[trig_matched_idx])
            eeg_act_datalist = [[], [], [], [] ,[], []] # RO, C1, C2, N1, N2, V order
            fnirs_act_datalist = [[], [], [], [] ,[], []] # RO, C1, C2, N1, N2, V order
            trig_key = ''
            if 'trigger' in this_patient_trig_file.keys():
                if len(this_patient_trig_file['trigger'][0].dtype) != 6: # AD s107
                    print('\n>> EEG Pass: trigger file does not have all acts ->', level, patient_name)
                    eeg_passed_patients.append(level + '_' + patient_name)
                    # print(this_patient_trig_file['trigger'][0].dtype)
                    continue
                trig_key = 'trigger'

            elif 'eeg' in this_patient_trig_file.keys():
                # print(this_patient_trig_file['eeg'][0].dtype)
                if len(this_patient_trig_file['eeg'][0].dtype) != 6:
                    print('\n>> EEG Pass: trigger file does not have all acts ->', level, patient_name)
                    eeg_passed_patients.append(level + '_' + patient_name)
                    # print(this_patient_trig_file['eeg'][0].dtype)
                    continue
                trig_key = 'eeg'
            
            pass_flag = False # True if verbal has no 'target'
            for f in fn: # 6 act
                act_name = f.replace('\\', '/').split('/')[-1]
                act_name = act_name.split('_')[-1].split('.')[0]
                eeg_act_file = io.loadmat(f)
                # print(level, patient_name)

                # Find corresponding fnirs file
                this_fnirs_file = '/'.join(f.replace('\\', '/').split('/')[:-2])
                this_fnirs_file = this_fnirs_file + '/fNIRs/' + eeg_fnirs_act_matching[act_name] +'.mat'
                fnirs_act_file = io.loadmat(this_fnirs_file)

                #################################################
                # print(f) # eeg file name
                # print(this_fnirs_file) # fnirs file name
                # print('*'*100)
                #################################################

                if act_name == 'RO': # save without slice
                    eeg_act_datalist[act_save_order[act_name]] = io.loadmat(f)['data']

                    fnirs_act_datalist[act_save_order[act_name]] = {}
                    fnirs_act_datalist[act_save_order[act_name]]['Hb'] = fnirs_act_file['Hb']
                    fnirs_act_datalist[act_save_order[act_name]]['HbO'] = fnirs_act_file['HbO']
                    fnirs_act_datalist[act_save_order[act_name]]['THb'] = fnirs_act_file['THb']
                    
                else: # C1, C2, N1, N2, V -> # save after slice
                    if trig_key == 'trigger':
                        sliced_eeg_data, sliced_fnirs_dict, well_done_flag = trigger_named_trigger(act_name=act_name, trig_file=this_patient_trig_file, eeg_mat=eeg_act_file, fnirs_mat=fnirs_act_file)
                        # print(sliced_eeg_data.shape)
                        eeg_act_datalist[act_save_order[act_name]] = sliced_eeg_data
                        fnirs_act_datalist[act_save_order[act_name]] = sliced_fnirs_dict

                    elif trig_key == 'eeg':
                        sliced_eeg_data, sliced_fnirs_dict, well_done_flag = trigger_named_eeg(act_name=act_name, trig_file=this_patient_trig_file, eeg_mat=eeg_act_file, fnirs_mat=fnirs_act_file)
                        if not well_done_flag: # if well_done_flag == False
                            print('\n>> EEG Pass: Verbal has no "target". ->', level, patient_name)
                            eeg_passed_patients.append(level + '_' + patient_name)
                            pass_flag = True
                            break
                        # print(sliced_eeg_data.shape)
                        eeg_act_datalist[act_save_order[act_name]] = sliced_eeg_data
                        fnirs_act_datalist[act_save_order[act_name]] = sliced_fnirs_dict

            
            if pass_flag: continue # pass if verbal has no 'target'

            all_len = [len(i) for i in eeg_act_datalist]
            # print(all_len)
            assert 0 not in all_len

            dat.append(eeg_act_datalist)
            mat.append(fnirs_act_datalist)
    

        data[i]['EEG'] = dat
        data[i]['fNIRs'] = mat


    # make save folder
    if not os.path.exists('./pickles'):
        os.makedirs('./pickles')

    save_file_name = 'eeg-' + str(global_slice_sec) + 'sec_fnirs-full_baselinecorrected_fnirs_sliced'

    # save all data pickle
    with open('./pickles/' + save_file_name + '.pkl', 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        print('>> ' + save_file_name + '.pkl saved.')

    # load all data pickle
    with open('./pickles/' + save_file_name + '.pkl', 'rb') as f:
        loaded_data = pickle.load(f) # ad, cn, ns, pre order
        print('>> ' + save_file_name + '.pkl loaded.')

    print('>> Total passed patients list:', eeg_passed_patients)

    return loaded_data



def segmentation_slice_base(dataset_root_path): # if __name__ == "__main__":
    
    # make data list
    eeg_path = (dataset_root_path + "Sorted_alzh_dataset/").replace('\\', '/')
    eeg_levels = ['AD', 'NORMAL', 'NP', 'MCI']
    # print(eeg_path)

    trig_pth = (dataset_root_path + 'Sorted_alzh_dataset/').replace('\\', '/')
    trig_levels = ['AD', 'NORMAL', 'NP', 'MCI']


    # # make pickle of all data
    # eeg_data_list = {'AD':[], 'NORMAL':[], 'NP':[], 'MCI':[]} # ad, cn, ns, pre order

    # for i, level in enumerate(eeg_levels):
    #     patient_list = glob(os.path.join(eeg_path, level, '*'))
    #     eeg_data_list[level] = [sorted(glob(os.path.join(x, 'EEG/*.mat'))) for x in patient_list]
    #     # data_list[i]['fNIRs'] = [sorted(glob(os.path.join(x, 'fNIRs/p_*.mat'))) for x in patient_list]
    # # print(eeg_data_list['AD'])


    # make trig list
    trig_list = {'AD':[], 'NORMAL':[], 'NP':[], 'MCI':[]}
    for i, level in enumerate(trig_levels):
        patient_list = glob(os.path.join(trig_pth, level, '*'))
        for x in patient_list:
            fold_list = os.listdir(x)
            
            # Trigger names: 'Trigger' or 'matdata' or 'Event extraction'
            if 'Trigger' in fold_list:
                file_list = glob(os.path.join(x, 'Trigger/*trig*.mat'))
                trig_list[eeg_levels[i]] += file_list
            elif 'matdata' in fold_list:
                file_list = glob(os.path.join(x, 'matdata/*trig*.mat'))
                trig_list[eeg_levels[i]] += file_list
            elif 'Event extraction' in fold_list:
                file_list = glob(os.path.join(x, 'Event extraction/*trig*.mat'))
                trig_list[eeg_levels[i]] += file_list
            else: raise
            
        # print(len(trig_list[eeg_levels[i]])) # 26, 67, 48, 48 -> finished!
    


    # print('*' * 100)
    print('>> AD patients num:', len(trig_list['AD'])) # 30
    print('>> Normal patients num:', len(trig_list['NORMAL'])) # 67 (70 - (patient 97, 99, 100))
    print('>> NP patients num:', len(trig_list['NP'])) # 50
    print('>> MCI patients num:', len(trig_list['MCI'])) # 49
    print('*' * 100)

    # preprocess
    sliced_data = make_sliced_data_pkl(eeg_levels, eeg_path, trig_list)

    # print(min(targets['C1']), max(targets['C1'])) # 20, 25
    # print(min(targets['C2']), max(targets['C2'])) # 20, 25
    # print(min(targets['N1']), max(targets['N1'])) # 20, 43
    # print(min(targets['N2']), max(targets['N2'])) # 14, 40

    # print(min(avg_target_terms['C1']), max(avg_target_terms['C1']))
    # print(min(avg_target_terms['C2']), max(avg_target_terms['C2']))
    # print(min(avg_target_terms['N1']), max(avg_target_terms['N1']))
    # print(min(avg_target_terms['N2']), max(avg_target_terms['N2']))

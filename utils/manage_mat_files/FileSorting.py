import os
import shutil
from distutils.dir_util import copy_tree # can copy to already exist folders

####### CHANGE HERE #########
root_path = "D:/치매감지/EEG_fNIRs_dataset/"
# fold_name = "Alzh_dataset_3rd_year/alzh_EEG_fNIRS/data/" # inside: only 'alzh, MCI, Np, normal' folders
# fold_name = "Alzh_dataset_3rd_year/alzh_2nd_year/치매감지/치매감지데이터/" # inside: only 'ad, cn, ns, pre' folders
fold_name = "Alzh_dataset_3rd_year/alzh_4th_year/"
#############################

origin_path = os.path.dirname(root_path + fold_name)
error_path = os.path.dirname(root_path + "Error/" + fold_name)
sorted_path = os.path.dirname(root_path + "Sorted_Dataset/Sorted_alzh_dataset/")

def CopyFiles(level_dict):
    print(level_dict)

    total_nums = {level_dict[level]:[0, 0, 0, 0] for level in os.listdir(origin_path)}
    eeg_allowed_format = ['RO.dat', 'C1.dat', 'C2.dat', 'N1.dat', 'N2.dat', 'V.dat'] # 'RO' = Resting Open. Necessary file name must be _RO.dat, _C1.dat, etc. 
    for origin_level in origin_levels:
        print('>> Origin level:', origin_level)
        new_level = level_dict[origin_level]
        print('>> New level:', new_level)
        
        level_fold = origin_path + '/' + origin_level
        if not os.path.isdir(level_fold): continue # pass if not folder
        
        
        for patient in os.listdir(level_fold):
            patient_name = patient[:4].lower() # S070 -> s070
            
            # EEG / fNIRs
            origin_patient_path = level_fold + '/' + patient
            new_level_patient_path = sorted_path + '/' + new_level + '/' + patient_name
            error_origin_level_patient_path = error_path + '/' + origin_level + '/' + patient
            for data_type in os.listdir(origin_patient_path):
                # EEG: (RO, RC, N1, N2, C1, C2, V)
                # file names: sS + patient num + Rnum + _ + type + .dat ex) sS057R01_RO.dat
                if data_type == 'EEG':
                    new_path_eeg = new_level_patient_path + '/EEG/'
                    total_file_list = [file for file in os.listdir(origin_patient_path + '/' + data_type)]
                    needed_file_list = sorted([file for file in os.listdir(origin_patient_path + '/' + data_type) if ('_' in file) and ('.dat' in file) and (file.split('_')[1] in eeg_allowed_format)])
                    if len(needed_file_list) != 6: # move entire patient folder to error
                        copy_tree(origin_patient_path, error_origin_level_patient_path)
                        break # pass this patient if # files < 6
                    if not os.path.exists(new_path_eeg):
                        os.makedirs(new_path_eeg)

                    # Copy necessary files to sorted folder
                    for file in needed_file_list:
                        origin_egg_file_path = origin_patient_path + '/' + data_type + '/' + file
                        new_egg_file_path = new_path_eeg + file
                        shutil.copy(origin_egg_file_path, new_egg_file_path) # copy
                        # print('>>', new_egg_file_path, ' saved.')
                        total_nums[new_level][1] += 1

                    # Copy the rest files to ERROR folder
                    remainders = list(set(total_file_list) - set(needed_file_list))
                    if remainders:
                        egg_error_path = error_origin_level_patient_path + '/' + data_type + '/'
                        if not os.path.exists(egg_error_path):
                            os.makedirs(egg_error_path)
                        for remain in remainders:
                            origin_egg_remain_path = origin_patient_path + '/' + data_type + '/' + remain
                            new_egg_remain_path = egg_error_path + remain
                            # print(new_egg_remain_path)
                            shutil.copy(origin_egg_remain_path, new_egg_remain_path) # copy

                    total_nums[new_level][0] += 1

                # fNIRs: (resting, oddball1, oddball2, nback1, nback2, verbal)
                # file names: p_ + num + _ + type + .mat ex) p_01_resting.mat
                elif data_type == 'fNIRs':
                    new_path_fnirs = new_level_patient_path + '/fNIRs/'
                    total_file_list = [file for file in os.listdir(origin_patient_path + '/' + data_type)]
                    needed_file_list = sorted([file for file in os.listdir(origin_patient_path + '/' + data_type) if 'p_' in file])[:6] # ~ verbal
                    if len(needed_file_list) != 6:
                        copy_tree(origin_patient_path, error_origin_level_patient_path)
                        break # pass this patient if # files < 6
                    if not os.path.exists(new_path_fnirs):
                        os.makedirs(new_path_fnirs)
                    
                    # Copy necessary files to sorted folder
                    for file in needed_file_list:
                        origin_fnirs_file_path = origin_patient_path + '/' + data_type + '/' + file
                        new_fnirs_file_path = new_path_fnirs + file
                        shutil.copy(origin_fnirs_file_path, new_fnirs_file_path) # copy
                        # print('>>', new_fnirs_file_path, ' saved.')
                        total_nums[new_level][3] += 1

                    # Copy the rest files to ERROR folder
                    remainders = list(set(total_file_list) - set(needed_file_list))
                    if remainders:
                        fnirs_error_path = error_origin_level_patient_path + '/' + data_type + '/'
                        if not os.path.exists(fnirs_error_path):
                            os.makedirs(fnirs_error_path)
                        for remain in remainders:
                            origin_fnirs_remain_path = origin_patient_path + '/' + data_type + '/' + remain
                            new_fnirs_remain_path = fnirs_error_path + remain
                            # print(new_fnirs_remain_path)
                            if os.path.isdir(origin_fnirs_remain_path): # if folder
                                copy_tree(origin_fnirs_remain_path, new_fnirs_remain_path)
                            else: # if files
                                shutil.copy(origin_fnirs_remain_path, new_fnirs_remain_path) # copy

                    total_nums[new_level][2] += 1

                else: # not 'EEG' or 'fNIRs'. ex) matdata
                    else_folder_path = origin_patient_path + '/' + data_type
                    else_error_path = error_origin_level_patient_path + '/' + data_type
                    if os.path.isdir(else_folder_path): # if data_type == directory
                        copy_tree(else_folder_path, else_error_path)
                    else: # if data_type == file name
                        if not os.path.exists(error_origin_level_patient_path + '/'):
                            os.makedirs(error_origin_level_patient_path + '/')
                        shutil.copy(else_folder_path, else_error_path)

        

    print(">> Total file nums [EEG, EEG_total, fNIRs, fNIRs_total]:", total_nums)

def PostProcessing(level_dict):
    # After copying files
    # Remove patient if # EEG != 6 or # fNIRs != 6
    # And count total num of files
    new_folder_total_num = {level_dict[level]:[0, 0, 0, 0] for level in os.listdir(origin_path)}
    for level in level_dict.values():
        level_fold = sorted_path + '/' + level        
        for patient in os.listdir(level_fold):
            patient_path = level_fold + '/' + patient        
            # EEG / fNIRs
            data_type_list = os.listdir(patient_path) # 'EEG' & 'fNIRs'
            if ('EEG' not in data_type_list) or ('fNIRs' not in data_type_list):
                # if EEG or fNIRs folder not made
                # => len != 6
                shutil.rmtree(patient_path)
                print('>>', patient_path, 'folder removed.')
                continue
            
            new_folder_total_num[level][0] += 1
            new_folder_total_num[level][1] += len([file for file in os.listdir(patient_path + '/EEG')])
            new_folder_total_num[level][2] += 1
            new_folder_total_num[level][3] += len([file for file in os.listdir(patient_path + '/fNIRs')])

    print(">> [SORTED] Total file nums [EEG, EEG_total, fNIRs, fNIRs_total]:", new_folder_total_num)

if __name__ == "__main__":
    # Standarizing levels
    # origin: ['alzh', 'MCI', 'N(p)', 'normal']
    # new_level:['AD', 'MCI', 'NP', 'NORMAL']
    origin_levels = [level for level in os.listdir(origin_path) if os.path.isdir(origin_path + '/' + level)]
    level_matching = {}
    for level in origin_levels:
        # print(level)
        if level[0].upper() == 'A':
            level_matching[level] = 'AD'
        elif (level[0].upper() == 'M') or (level.upper() == 'PRE'): # pre = 2nd_year's
            level_matching[level] = 'MCI'
        elif (level.upper() == 'N(P)') or (level.upper() == 'NS') or (level.upper() == 'NP'): # ns = 2nd_year's
            level_matching[level] = 'NP'
        elif (level == 'Normal') or (level == 'normal') or (level == '정상') or (level.upper() == 'CN'): # cn = 2nd_year's
            level_matching[level] = 'NORMAL'

    # RUN
    CopyFiles(level_matching)
    PostProcessing(level_matching) # Need to run after running CopyFiles

import os

fold_name = "Alzh_dataset_3rd_year/alzh_EEG_fNIRS/data/" # inside: alzh, MCI, Np, normal
origin_path = "C:/Users/rit/Desktop/석사_3학기/치매감지/" + fold_name
error_path = "C:/Users/rit/Desktop/석사_3학기/치매감지/Error"

# make EEG folders for each patient
######### Exception:
# C:\Users\rit\Desktop\석사_3학기\치매감지\Alzh_dataset_3rd_year\alzh_EEG_fNIRS\data\N(p)\s057\s057EEG\s057EEG
# -> C:\Users\rit\Desktop\석사_3학기\치매감지\Alzh_dataset_3rd_year\alzh_EEG_fNIRS\data\N(p)\s057\s057EEG
for (root, directories, files) in os.walk(origin_path):
    for d in directories:
            d_path = os.path.join(root, d)
            file_path = d_path.replace('\\', '/')
            if file_path[-3:] == 'EEG':
                splitted = file_path.split('/')
                # print(splitted[-6:]) # ['Alzh_dataset_3rd_year', ~ 'EEG']
                new_error_path = error_path + '/' + '/'.join(splitted[-6:])
                print(new_error_path)
                if not os.path.exists(new_error_path):
                    os.makedirs(new_error_path)
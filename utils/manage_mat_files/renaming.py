import os

level = 'Normal'
patient = 's292'
pth = 'D:/치매감지/EEG_fNIRs_dataset/Alzh_dataset_3rd_year/alzh_4th_year/' + level + '/' + patient + '/EEG/'
# types = ['_RO', '_RO', '_C1', '_C1', '_C2', '_C2', '_N1', '_N1', '_N2', '_N2', '_V', '_V']
# types = ['_RO', '_N1', '_N2', '_C1', '_C2', '_V']
types = ['_RO', '_C1', '_C2', '_N1', '_N2', '_V']

i = 0
for file in os.listdir(pth):
    if ('dat' not in file) and ('mat' not in file): continue
    temp = file.split('.')
    origin = pth + file
    renamed = ''
    if temp[1] == 'dat':
        renamed = pth + temp[0] + types[i] + '.dat'
    elif temp[1] == 'mat':
        renamed = pth + temp[0] + types[i] + '.mat'
    print(origin)
    print(renamed)
    os.rename(origin, renamed)
    i += 1

    if i == 6: break


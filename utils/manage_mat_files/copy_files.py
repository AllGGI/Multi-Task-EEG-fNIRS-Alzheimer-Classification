import os
import shutil

path = os.path.dirname("D:/치매감지/Wearable_dataset")
from_path = os.path.dirname("E:/taemin/")
levels = ['AD', 'MCI(p)', '정상', 'N(p)']

for level in levels:
    save_path = path + '/' + level
    from_path2 = from_path + '/' + level
        
    for (root, directories, files) in os.walk(from_path2):
        for file in files:
            file_path = os.path.join(root, file)
            # if (('Tetrax' in file_path) or ('tetrax' in file_path) or 
            #     ('EEG' in file_path) or ('fNIRS' in file_path) or ('fNIRs' in file_path) or
            #     ('Camcorder' in file_path) or ('CCTV' in file_path) or ('footlogger' in file_path) or
            #     ('Kinect' in file_path) or ('cctv' in file_path) or ('camcorder' in file_path)
            #     ): continue
            # if ('Wearable' not in file_path) and ('wearable' not in file_path): print(file_path)
            key = ''
            file_path = file_path.replace('\\', '/')
            if ('Wearable' in file_path): key = 'Wearable'
            elif ('wearable' in file_path): key = 'wearable'
            elif ('행동' in file_path):
                key = file_path.split('/')[-1]

            if key == '': continue

            splitted = file_path.split('/')
            splitted[0] = 'D:/치매감지'
            splitted[1] = 'Wearable_dataset'
            new_file_name = '/'.join(splitted)
            new_path = '/'.join(splitted[:-1])

            if not os.path.exists(new_path):
                os.makedirs(new_path)
            shutil.copy(file_path, new_file_name)

            # if ('뇌' in file_path) and file_path[-4:] == '.pdf':
            #     file_path = file_path.replace('\\', '/')
            #     splitted = file_path.split('/')
            #     splitted[0] = 'C:/Users/rit/Desktop/석사_3학기/치매감지'
            #     splitted[1] = '세부사항'
            #     new_file_name = '/'.join(splitted)
            #     new_path = '/'.join(splitted[:-1])
            #     if not os.path.exists(new_path):
            #         os.makedirs(new_path)
            #     shutil.copy(file_path, new_file_name)
                


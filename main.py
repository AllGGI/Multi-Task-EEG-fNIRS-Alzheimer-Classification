import os

# My files
from config import get_config
from P1_v1_base_slice_sec import segmentation_slice_base
from P1_v1_extraAD_slice_sec import segmentation_slice_extra
from P1_v2_base_slice_full import segmentation_full_base
from P1_v2_extraAD_full import segmentation_full_extra
from P1_v3_base_fnirs_timedomain_slice import segmentation_timedomain_base
from P1_v3_extraAD_fnirs_timedomain_slice import segmentation_timedomain_extra

from P2_v1_PSD_feature_extraction import psd_feature_extraction
from P2_v1_PSD_to_csv import psd2csv
from P2_v2_fnirs_timedomain_to_csv import timedomain2csv
from P3_EEGfNIRS_multimodal_to_csv import multimodal2csv # EEG + fNIRS csv

from P4_v2_RFECV_feature_selection import rfecv_feature_selection, classification

from P5_v1_previous_study import Comparison1 # class
# usage: comp1 = Comparison1(slice_sec_num=10)
#        comp1.run()

from P5_v2_comparison_of_extraction_method import extraction_comparison2csv
from P5_v3_comparison_of_selection_method import PrevStudy_feature_selection, PrevStudy_test


def main(args):

    # mode: [segmentation / extraction / selection / classification]

    # RUN) python main.py --mode segmentation
    if args.mode == "segmentation":
        # make pickles from EEG/fNIRS .mat files
        segmentation_slice_base(args.data_root)
        segmentation_slice_extra(args.data_root)
        segmentation_full_base(args.data_root)
        segmentation_full_extra(args.data_root)
        segmentation_timedomain_base(args.data_root)
        segmentation_timedomain_extra(args.data_root)
        print('>> Segmentation finished.')


    # RUN) python main.py --mode extraction
    elif args.mode == "extraction":
        psd_feature_extraction()
        psd2csv()
        timedomain2csv()
        multimodal2csv() # EEG + fNIRS csv

    # RUN) python main.py --mode selection --exp 1 --task R 
    elif args.mode == "selection":
        task_dict = {'R': [0], 'C': [1,2], 'N': [3,4], 'V': [5]}
        csv_root = './csv_folder/Experiment' + str(args.exp) + '/'
        save_root = './Experimental_results/Experiment' + str(args.exp) + '/'

        if args.exp == 1:
            task_list = (args.task).split(',')
            act_num = []
            for t in task_list: act_num += task_dict[t]
            
            pth = '5secEEGPSD_FullFnirsPSD_FullFnirsTimeDomain'
            rfecv_feature_selection(args.cv_num, act_num, pth, csv_root, save_root, args.seed)
            
        elif args.exp == 2:
            act_num = [0,1,2,3,4,5]

            # [Exp2-A: EEG + fNIRS | Exp2-B: EEG | Exp2-C: fNIRS]
            pths = ['5secEEGPSD_FullFnirsPSD_FullFnirsTimeDomain',
                    'opt-2_5secEEGPSD',
                    'opt-3_FullFnirsPSD_FullFnirsTimeDomain']
            
            for pth in pths: rfecv_feature_selection(args.cv_num, act_num, pth, csv_root, save_root, args.seed)

        
        elif args.exp == 3:
            act_num = [0,1,2,3,4,5]

            # Compare feature extraction methods (Ours vs Prev)
            # [Exp3-A: MyEEG + MyfNIRS + RFECV, Exp3-B: MyEEG+PrevfNIRS+RFECV, Exp3-C: PrevEEG+MyfNIRS+RFECV, Exp3-D: PrevEEG+PrevfNIRS+RFECV]
            extraction_comparison2csv(act_num) # make each csv
            pths = ['5secEEGPSD_FullFnirsPSD_FullFnirsTimeDomain', 'MyEEG_PrevfNIRS', 'PrevEEG_MyfNIRS', 'PrevEEG_PrevfNIRS']
            for pth in pths: rfecv_feature_selection(args.cv_num, act_num, pth, csv_root, save_root, args.seed)

            # Compare feature selection methods (Ours(RFECV) vs Prev(PCCFS))
            # Exp3-E: PrevEEG + PrevfNIRS + PCCFS
            PrevStudy_feature_selection(csv_root)

    
    # RUN) python main.py --mode classification --exp 1 --task R
    elif args.mode == "classification":
        task_dict = {'R': [0], 'C': [1,2], 'N': [3,4], 'V': [5]}
        save_root = './Experimental_results/Experiment' + str(args.exp) + '/'
        csv_root = './csv_folder/Experiment' + str(args.exp) + '/'

        if args.exp == 1:
            print('>> Experiment 1:') # Exp1 with config defined tasks
            task_list = (args.task).split(',')
            act_num = []
            for t in task_list: act_num += task_dict[t]
            
            pth = '5secEEGPSD_FullFnirsPSD_FullFnirsTimeDomain'
            classification(args.cv_num, act_num, pth, csv_root, save_root, args.seed, args.clf_type)
            
        elif args.exp == 2:
            print('>> Experiment 2:') # [Exp2-A: EEG + fNIRS | Exp2-B: EEG | Exp2-C: fNIRS]
            act_num = [0,1,2,3,4,5]
            pths = ['5secEEGPSD_FullFnirsPSD_FullFnirsTimeDomain',
                    'opt-2_5secEEGPSD',
                    'opt-3_FullFnirsPSD_FullFnirsTimeDomain']
            
            for pth in pths: classification(args.cv_num, act_num, pth, csv_root, save_root, args.seed, args.clf_type)

        elif args.exp == 3:
            act_num = [0,1,2,3,4,5]

            # Compare feature extraction methods (Ours vs Prev)
            # [Exp3-A: MyEEG + MyfNIRS + RFECV, Exp3-B: MyEEG+PrevfNIRS+RFECV, Exp3-C: PrevEEG+MyfNIRS+RFECV, Exp3-D: PrevEEG+PrevfNIRS+RFECV]
            pths = ['5secEEGPSD_FullFnirsPSD_FullFnirsTimeDomain', 'MyEEG_PrevfNIRS', 'PrevEEG_MyfNIRS', 'PrevEEG_PrevfNIRS']
            idxs = {0:'A', 1:'B', 2:'C', 3:'D'}
            for i, pth in enumerate(pths):
                print('='*100)
                print('>> Experiment 3-' + idxs[i] + ':')
                classification(args.cv_num, act_num, pth, csv_root, save_root, args.seed, args.clf_type)
                print()

            # Compare feature selection methods (Ours(RFECV) vs Prev(PCCFS))
            # Exp3-E: PrevEEG + PrevfNIRS + PCCFS
            print('='*100)
            print('>> Experiment 3-E:')
            PrevStudy_test(csv_root, save_root)


    else:
        print('>> Please execute with mode: ex) --mode "selection"')
        print('>> MODE: [segmentation / extraction / selection / classification]')


if __name__ == "__main__":

    config, unparsed = get_config()
    # run_time = date.today().strftime("%m-%d") + datetime.now().strftime("-%H-%M")
    # save_dir = config.save_path + run_time + "/"

    # store configuration
    with open("./config.txt", "w") as f:
        f.write(">> YOUR parameters for " + config.mode + ":\n\n")
        for arg in vars(config):
            argname = arg
            contents = str(getattr(config, arg))
            f.write(argname + " = " + contents + "\n")

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu

    # print(config)
    main(config)


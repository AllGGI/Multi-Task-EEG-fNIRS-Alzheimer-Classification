# Configuration file for (base training / adversarial training / adversarial attack)

import argparse
from distutils import util

# Basic Configuration
parser = argparse.ArgumentParser(description="Base Training")


# configs
parser.add_argument("--data_root", default="E:\최종_치매감지백업\데이터셋\EEG_fNIRs_dataset\Sorted_Dataset_234/", type=str)
parser.add_argument("--gpu", default="0,1,2,3", type=str, help="GPU id to use.")
parser.add_argument(
    "--mode",
    type=str,
    default='classification',
    help="Type of execution modes: [segmentation / extraction / selection / classification(default)]",
)
parser.add_argument(
    "--exp",
    type=int,
    default=1,
    help="Experiment number: [1 - Tasks / 2 - Modals / 3 - Previous study]",
)
parser.add_argument(
    "--task",
    type=str,
    default='R,C,N,V',
    help="Only for Experiment1 (default='R,C,N,V'). Lists of tasks: R(Resting), C(Oddball), N(1-back), V(Verbal fluency)",
)
parser.add_argument(
    "--seed",
    type=int,
    default=1263,
    help="Random seed number.",
)
parser.add_argument(
    "--cv_num",
    type=int,
    default=5,
    help="The number of cross-validation folds.",
)
parser.add_argument(
    "--clf_type",
    type=str,
    default='Tree',
    help="Type of classification model: [Tree(default) | SVM | RF | MLP]",
)





def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed


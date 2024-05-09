"""
Add train/val/test set shortcuts to HDF5 data file for FC matrices.
Creates those shortcuts for the defined split + variants of interest.

Run e.g. with:
$ python3 add_split_shortcuts.py --name "100-500p" --datasplit_dir "../../data/schaefer/" -- variants "7n100p" "7n200p" "7n500p" "17n100p" "17n200p" "17n500p"
"""

import numpy as np
import h5py
import argparse
from brain_age_prediction import sklearn_utils


##################################################################################
# prepare data paths
save_dir = '/ritter/share/projects/laura_riedel_thesis/'

# define variants of interest
variants = ['7n100p','7n200p','7n500p','17n100p','17n200p','17n500p']

# get parameters from console
parser = argparse.ArgumentParser(prog='AddSplitShortcut',
                                description='Add train/val/test shortcuts to HDF5 File containint FC matrices for easier access.',
                                epilog='Shortcut added.')

parser.add_argument('--name', type=str, default='100-500p',
                    help='Enter directory name to save under.')
parser.add_argument('--datasplit_dir', type=str, default='../../data/schaefer/', 
                    help='Enter path to where the "data_info/" directory is located. This gives information which splits to use (and which IDs to ignore based on the related heldout test set). Default: ../../data/schaefer/')
parser.add_argument('--variants', nargs='+', default=variants, 
                    help='Enter variants of interest. Default: "7n100p" "7n200p" "7n500p" "17n100p" "17n200p" "17n500p"')

args = parser.parse_args()

##################################################################################
print('VARIANTS:', args.variants)

assert set(args.variants).intersection(set(variants)) == set(args.variants), "There is a mistake in the variable names. Check the spelling!"

# create shortcuts for all defined variables
with h5py.File(save_dir+'schaefer_fc_matrices.hdf5', 'a') as hdf5:
    grp = hdf5.create_group('split_shortcuts/'+args.name)
    for variant in args.variants:
        print(f'Create shortcut for variant {variant}...')
        X_train, y_train, X_val, y_val, X_test, y_test = sklearn_utils.load_dataset(variant, args.datasplit_dir)
        grp.create_dataset(variant+'/train/X', data=X_train, compression='gzip', compression_opts=9)
        grp.create_dataset(variant+'/train/y', data=y_train, compression='gzip', compression_opts=9)
        grp.create_dataset(variant+'/val/X', data=X_val, compression='gzip', compression_opts=9)
        grp.create_dataset(variant+'/val/y', data=y_val, compression='gzip', compression_opts=9)
        grp.create_dataset(variant+'/test/X', data=X_test, compression='gzip', compression_opts=9)
        grp.create_dataset(variant+'/test/y', data=y_test, compression='gzip', compression_opts=9)
        print('Done.')
        
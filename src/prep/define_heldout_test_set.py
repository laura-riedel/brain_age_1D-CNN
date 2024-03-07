"""
Sample subject IDs randomly to use as held-out final test set and save the list.

Call with e.g.:
$ python3 define_heldout_test_set.py --k 5000 --seed 43 --name "heldout_test_set.csv"
"""
import pandas as pd
import random
import argparse
import csv
from brain_age_prediction import utils

# prepare data path
schaefer_data_dir = '../../data/schaefer/'

##################################################################################
# get parameters from console
parser = argparse.ArgumentParser(prog='DefineHeldoutTestSet',
                                description='Sample subject IDs randomly to use as held-out final test set and save the list.',
                                epilog='Sampling complete, list saved.')

parser.add_argument('--k', type=int, default=5000, help='Enter number of requested ID samples. Default: 5000')
parser.add_argument('--seed', type=int, default=43, help='Enter random seed to be applied. Default: 43')
parser.add_argument('--name', type=str, default='heldout_test_set.csv', help='Enter file name to save as. Default: "heldout_test_set.csv"')

args = parser.parse_args()

##################################################################################
# make random sampling reproducible
utils.make_reproducible(args.seed)

# load availability overview
schaefer_exists_df = pd.read_csv(schaefer_data_dir+'schaefer_exists.csv')
# limit to cases of existing, non-empty files
schaefer_exists_df.drop(schaefer_exists_df[schaefer_exists_df['schaefer_exists'] == False].index, inplace=True)
schaefer_exists_df.drop(schaefer_exists_df[schaefer_exists_df['is_empty'] == True].index, inplace=True)
schaefer_exists_df.drop(schaefer_exists_df[schaefer_exists_df['contains_nan'] == True].index, inplace=True)
# reset index
schaefer_exists_df = schaefer_exists_df.reset_index(drop=True, inplace=False)

# get list of subject IDs
sub_ids = list(schaefer_exists_df['eid'])     

# sample for held-out test set
test_set = random.sample(sub_ids, k=args.k)

# save
with open(schaefer_data_dir+args.name, 'w', newline='') as f:
    writer = csv.writer(f)
    for sub in test_set:
        writer.writerow([sub])
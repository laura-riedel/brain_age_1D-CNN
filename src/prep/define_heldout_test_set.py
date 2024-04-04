"""
Sample subject IDs randomly to use as held-out final test set and save the list.

Call with e.g.:
$ python3 define_heldout_test_set.py --k 5000 --seed 43 --name "heldout_test_set.csv" --variants '7n100p' '7n200p' '7n500p' '17n100p' '17n200p' '17n500p'
"""
import pandas as pd
import random
import argparse
import csv
from brain_age_prediction import utils

# prepare data path
schaefer_data_dir = '../../data/schaefer/'
# baseline variants
all_variants = ['7n100p','7n200p','7n500p','7n700p','7n1000p','17n100p','17n200p','17n500p','17n700p','17n1000p']

##################################################################################
# get parameters from console
parser = argparse.ArgumentParser(prog='DefineHeldoutTestSet',
                                description='Sample subject IDs randomly to use as held-out final test set and save the list.',
                                epilog='Sampling complete, list saved.')

parser.add_argument('--k', type=int, default=5000, 
                    help='Enter number of requested ID samples. Default: 5000')
parser.add_argument('--seed', type=int, default=43, 
                    help='Enter random seed to be applied. Default: 43')
parser.add_argument('--name', type=str, default='heldout_test_set.csv', 
                    help='Enter file name to save as. Default: "heldout_test_set.csv"')
parser.add_argument('--variants', nargs='+', default=all_variants, 
                    help='Enter variants of interest. Default: "7n100p" "7n200p" "7n500p" "7n700p" "7n1000p" "17n100p" "17n200p" "17n500p" "17n700p" "17n1000p"')

args = parser.parse_args()

##################################################################################
# make random sampling reproducible
utils.make_reproducible(args.seed)

print('VARIANTS:', args.variants)

assert set(args.variants).intersection(set(all_variants)) == set(args.variants), "There is a mistake in the variable names. Check the spelling!"

# get list of subject IDs
sub_ids = utils.get_usable_schaefer_ids(variants=args.variants)

# sample for held-out test set
test_set = random.sample(sub_ids, k=args.k)

# save
with open(schaefer_data_dir+args.name, 'w', newline='') as f:
    writer = csv.writer(f)
    for sub in test_set:
        writer.writerow([sub])
# CREATE OVERVIEW: for which subs do Schaefer timeseries exist?
"""
This script iterates through all subjects within the saved UK BioBank BIDS directory (see ukbb_dir) and checks 
    a) whether files of Schaefer parcellation rs-fMRI timeseries exist for this subject and
    b) whether the existing files actually contain timeseries content or are empty and
    c) whether the non-empty files contain NaN values or timeseries containing only 0s (and where)
for all Schaefer parcellation variants.

Two overviews are created per Schaefer variant in the specified directory (schaefer_data_dir/VARIANT):
- `schaefer_exists.csv`: 6 columns (eid, schaefer_exists, is_empty, contains_nan, contains_0) which states  
                        for each subject ID if files exist (schaefer_exists -> Bool), 
                        whether they are empty (is_empty -> Bool) and
                        whether they contain NaN values (contains_nan -> Bool) 
                        or only 0's (contains_0 -> Bool) and where (location_of_0 -> List(Int)).
- `empty_files.csv`: simple collection of the subject IDs containing empty Schaefer files
"""
# import packages
import pandas as pd
import numpy as np
from scipy.stats import zscore
from pathlib import Path
import os
import csv

# prepare data paths
ukbb_dir = '/ritter/share/data/UKBB/ukb_data/bids/'
schaefer_data_dir = '../../data/schaefer/'

# define existing variants
variants = ['7n100p','7n200p','7n500p','7n700p','7n1000p',
            '17n100p','17n200p','17n500p','17n700p','17n1000p']

# ID extraction
def strip_id(string):
    """Function extracts a subject's ID from the string 'sub-[ID]' and returns the ID."""
    return string.replace('sub-','')    

def fill_in_defaults(i):
    schaefer_exists_df.loc[i, 'is_empty'] = True
    schaefer_exists_df.loc[i, 'contains_nan'] = False
    schaefer_exists_df.loc[i, 'contains_0'] = False
    schaefer_exists_df.loc[i, 'location_of_0'] = str([])

# getting all subjects
subs = [f.name for f in os.scandir(ukbb_dir) if f.is_dir()]

# iterate through all Schaefer parcellation variants
for variant in variants:
    print('Schaefer variant:', variant)
    # prepare df for collection
    schaefer_exists_df = pd.DataFrame(columns=['eid','schaefer_exists','is_empty', 'contains_nan', 'contains_0', 'location_of_0'])

    # take note of empty files
    exceptions = []

    # find out for which eid timeseries data exist for this Schaefer parcellation
    i = 0
    for sub in subs:
        sub_id = strip_id(sub)
        schaefer_exists_df.loc[i, 'eid'] = sub_id
        sub_dir = ukbb_dir+sub+'/ses-2/func/'+sub+'_ses-2_task-rest_Schaefer'+variant+'.csv.gz'
        if os.path.exists(sub_dir):
            schaefer_exists_df.loc[i, 'schaefer_exists'] = True
            # check if file is empty
            if os.path.getsize(sub_dir) > 0:
                # if not empty
                schaefer_exists_df.loc[i, 'is_empty'] = False
                timeseries = np.genfromtxt(sub_dir, skip_header=1, usecols=tuple([i for i in range(1,491)]), delimiter=',') 
                # check for NaN values, update column if True
                if np.isnan(timeseries).any():
                    schaefer_exists_df.loc[i, 'contains_nan'] = True
                else:
                    schaefer_exists_df.loc[i, 'contains_nan'] = False
                # check for rows only containing 0s
                zero_rows_idx = list(np.where(~timeseries.any(axis=1))[0])
                if zero_rows_idx:
                    schaefer_exists_df.loc[i, 'contains_0'] = True
                    schaefer_exists_df.loc[i, 'location_of_0'] = str(zero_rows_idx)
                else:
                    schaefer_exists_df.loc[i, 'contains_0'] = False
                    schaefer_exists_df.loc[i, 'location_of_0'] = str([])
            else:
                # if empty
                exceptions.append(sub_id)
                fill_in_defaults(i)
        else:
            schaefer_exists_df.loc[i, 'schaefer_exists'] = False
            fill_in_defaults(i)
        i += 1
        
    # save
    # create directory if necessary
    Path(schaefer_data_dir+variant+'/').mkdir(parents=True, exist_ok=True)
    # overview existing + (non-)empty files
    schaefer_exists_df.to_csv(schaefer_data_dir+variant+'/schaefer_exists.csv', index=False)
    # overivew empty files
    with open(schaefer_data_dir+variant+'/empty_files.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for sub in exceptions:
            writer.writerow([sub])
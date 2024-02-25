# CREATE OVERVIEW: for which subs do Schaefer timeseries exist?
"""
This script iterates through all subjects within the saved UK BioBank BIDS directory (see ukbb_dir) and checks 
a) whether files of Schaefer parcellation rs-fMRI timeseries exist for this subject and
b) whether the existing files actually contain timeseries content or are empty.

Two overviews are created in the specified directory (schaefer_data_dir):
- `schaefer_exists.csv`: 3 columns (eid, schaefer_exists, is_empty) which states for each subject ID 
                        if files exist (schaefer_exists -> Bool) and whether they are empty (is_empty -> Bool)
- `empty_files.csv`: simple collection of the subject IDs containing empty Schaefer files
"""
# import packages
import pandas as pd
import os
import csv

# prepare data paths
ukbb_dir = '/ritter/share/data/UKBB/ukb_data/bids/'
schaefer_data_dir = '../../data/schaefer/'

# ID extraction
def strip_id(string):
    """Function extracts a subject's ID from the string 'sub-[ID]' and returns the ID."""
    return string.replace('sub-','')    

# getting all subjects
subs = [f.name for f in os.scandir(ukbb_dir) if f.is_dir()]

# prepare df for collection
schaefer_exists_df = pd.DataFrame(columns=['eid','schaefer_exists','is_empty'])

# take note of empty files
exceptions = []

# find out for which eid Schaefer parcellation timeseries data exist
i = 0
for sub in subs:
    sub_id = strip_id(sub)
    schaefer_exists_df.loc[i, 'eid'] = sub_id
    sub_dir = data_dir+sub+'/ses-2/func/'+sub+'_ses-2_task-rest_Schaefer7n100p.csv.gz'
    if os.path.exists(sub_dir):
        schaefer_exists_df.loc[schaefer_exists_df['eid'] == sub_id, 'schaefer_exists'] = True
        # check if file is empty
        if os.path.getsize(sub_dir) > 0:
            # if not empty
            schaefer_exists_df.loc[schaefer_exists_df['eid'] == sub_id, 'is_empty'] = False
        else:
            # if empty
            exceptions.append(sub_id)
            schaefer_exists_df.loc[schaefer_exists_df['eid'] == sub_id, 'is_empty'] = True
    else:
        schaefer_exists_df.loc[schaefer_exists_df['eid'] == sub_id, 'schaefer_exists'] = False
    i += 1
        
# save
# overview existing + (non-)empty files
schaefer_exists_df.to_csv(schaefer_data_dir+'schaefer_exists.csv', index=False)
# overivew empty files
with open(schaefer_data_dir+'empty_files.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    for sub in exceptions:
        writer.writerow([sub])
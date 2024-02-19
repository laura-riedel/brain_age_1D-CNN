# CREATE OVERVIEW: for which subs do Schaefer timeseries exist?
# import packages
import pandas as pd
import os

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
schaefer_exists_df = pd.DataFrame(columns=['eid','schaefer_exists'])

# find out for which eid Schaefer parcellation timeseries data exist
i = 0
for sub in subs:
    sub_id = strip_id(sub)
    schaefer_exists_df.loc[i, 'eid'] = sub_id
    if os.path.exists(data_dir+sub+'/ses-2/func/'+sub+'_ses-2_task-rest_Schaefer7n100p.csv.gz'):
        schaefer_exists_df.loc[schaefer_exists_df['eid'] == sub_id, 'schaefer_exists'] = True
    else:
        schaefer_exists_df.loc[schaefer_exists_df['eid'] == sub_id, 'schaefer_exists'] = False
    i += 1
        
# save
schaefer_exists_df.to_csv(schaefer_data_dir+'schaefer_exists.csv', index=False)
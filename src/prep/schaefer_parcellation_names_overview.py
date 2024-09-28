"""
Script that creates a overviews of network names:
For each of the chosen Schaefer variants, it creates an overview 
which networks the parcellations belong to.

Run e.g. with:
$ python3 schaefer_parcellation_names_overview.py
"""
# import packages
import pandas as pd

# prepare data paths
ukbb_dir = '/ritter/share/data/UKBB/ukb_data/bids/'
schaefer_data_dir = '../../data/schaefer/'

# define existing variants
variants = ['7n100p','7n200p','7n500p','7n700p','7n1000p',
            '17n100p','17n200p','17n500p','17n700p','17n1000p']

# create separate overviews for each network/parcellation combination
for variant in variants:
    ts_path = ukbb_dir+'sub-1000014/ses-2/func/sub-1000014_ses-2_task-rest_Schaefer'+variant+'.csv.gz'
    ts_df = pd.read_csv(ts_path)
    # save
    ts_df['label_name'].to_csv(schaefer_data_dir+variant+'/label_names_'+variant+'.csv', header=False, index=False)
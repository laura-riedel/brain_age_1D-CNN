"""
Script that creates an overview for each Schaefer variant:
Per variant, which parcellations contain only 0-timeseries, and how often?

Run e.g. with:
$ python3 schaefer_zero_ts_overview.py
"""
# import packages
import pandas as pd
from collections import Counter
import re

# prepare data path
schaefer_data_dir = '../../data/schaefer/'

# define existing variants
variants = ['7n100p','7n200p','7n500p','7n700p','7n1000p',
            '17n100p','17n200p','17n500p','17n700p','17n1000p']

##################################################################################
def count_0_occurrences(df):
    """
    For subjects that have existing Schaefer files whithout missing values,
    which parcellation timeseries contain only zeros?
    Counts how often parcellations themselves are zeroed.
    """
    existence = (df['schaefer_exists'] == True)
    emptiness = (df['is_empty'] == False)
    nans = (df['contains_nan'] == False)
    zeros = (df['contains_0'] == True)
    condition = existence & emptiness & nans & zeros
    parcellations = []
    for i in df[condition]['location_of_0']:
        numbers = [int(number) for number in re.findall('\d+',i)]
        parcellations += numbers
    return Counter(parcellations) 

##################################################################################
for variant in variants:
    full_path = schaefer_data_dir+'/'+variant+'/'
    try:
        schaefer_exists_df = pd.read_csv(full_path+'schaefer_exists.csv')
    except:
        print('FILE DOES NOT EXIST: schaefer_exists.csv for variant', variant)
        print('You might need to run the script "schaefer_accessible_data_overview.py" first!')
        sys.exit()
    # extract + count zero-parcellation occurrences, create df
    count_dict = count_0_occurrences(schaefer_exists_df)
    count_df = pd.DataFrame.from_dict(count_dict, orient='index', columns=['zero_count'])
    # merge counts with network labels into overview
    label_df = pd.read_csv(full_path+'label_names_'+variant+'.csv', names=['network_name'])
    overview_df = pd.concat([label_df,count_df], axis=1, join='inner')
    # save
    overview_df.to_csv(full_path+'zero_ts_overview.csv')
    
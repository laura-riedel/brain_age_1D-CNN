"""
Iterate through all known UKBB subjects. If they have Schaefer parcellation timeseries,
calculate the functional connectivity (FC) matrix and save the upper triangle as HDF5 file.
"""

import numpy as np
import h5py

from brain_age_prediction import utils

##################################################################################
# prepare data paths
ukbb_dir = '/ritter/share/data/UKBB/ukb_data/bids/'
save_dir = '/ritter/share/projects/laura_riedel_thesis/'

# define custom functions
def get_fc_matrix(sub_id, variant):
    """
    Loads timeseries for current subject ID + Schaefer parcellation variant and
    calculates that timeseries' FC matrix.
    Input: 
        sub_id: subject ID (as int, e.g. 1234567)
        variant: Schaefer parcellation variant (as string, e.g. '7n100p')
    Output:
        functional connectivity matrix
    """
    ts_path = ukbb_dir+'sub-'+str(sub_id)+'/ses-2/func/sub-'+str(sub_id)+'_ses-2_task-rest_Schaefer'+variant+'.csv.gz'
    ts = np.loadtxt(ts_path, skiprows=1, usecols=tuple([i for i in range(1,491)]), delimiter=',')
    return np.corrcoef(ts, dtype='float32')

def create_hdf5_datasets(hdf5_group, sub_id, variant):
    """
    Calls get_fc_matrix and then saves the upper triangle of the FC matrix to the specified HDF5 group.
    Input:
        hdf5_group: previously determined HDF5 Group object to save the matrices to
        sub_id: subject ID for which to calculate the FC matrices
        variant: Schaefer parcellation variant for which to calculate the FC matrices
    """
    fc_matrix = get_fc_matrix(sub_id, variant)
    hdf5_group.create_dataset(variant+'_triu', data=np.triu(fc_matrix, 1), compression='gzip', compression_opts=9)

##################################################################################


# get overview of all possible subject IDs
all_sub_ids = utils.get_schaefer_overview(variants=['7n100p'])['eid'].values
# get lists of usable subject IDs
ids_7n100p = utils.get_usable_schaefer_ids(variants=['7n100p'])
ids_7n200p = utils.get_usable_schaefer_ids(variants=['7n200p'])
ids_7n500p = utils.get_usable_schaefer_ids(variants=['7n500p'])
ids_17n100p = utils.get_usable_schaefer_ids(variants=['17n100p'])
ids_17n200p = utils.get_usable_schaefer_ids(variants=['17n200p'])
ids_17n500p = utils.get_usable_schaefer_ids(variants=['17n500p'])

# find unusable subject IDs
unusable_ids = set(all_sub_ids)
for parcellation in [set(ids_7n100p), set(ids_7n200p), set(ids_7n500p), 
                     set(ids_17n100p), set(ids_17n200p), set(ids_17n500p)]:
    unusable_ids = unusable_ids-parcellation

with h5py.File(save_dir+'schaefer_fc_matrices.hdf5', 'w') as hdf5:
    i = 1
    for sub in all_sub_ids:
        print(f'Subject {i}/{len(all_sub_ids)}')
        if sub not in unusable_ids:
            grp = hdf5.create_group('bids/'+str(sub))
            if sub in ids_7n100p:
                variant = '7n100p'
                create_hdf5_datasets(grp, sub, variant)
            if sub in ids_7n200p:
                variant = '7n200p'
                create_hdf5_datasets(grp, sub, variant)
            if sub in ids_7n500p:
                variant = '7n500p'
                create_hdf5_datasets(grp, sub, variant)
            if sub in ids_17n100p:
                variant = '17n100p'
                create_hdf5_datasets(grp, sub, variant)
            if sub in ids_17n200p:
                variant = '17n200p'
                create_hdf5_datasets(grp, sub, variant)
            if sub in ids_17n500p:
                variant = '17n500p'
                create_hdf5_datasets(grp, sub, variant)
        i += 1
    
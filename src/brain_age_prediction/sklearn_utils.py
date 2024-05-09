import numpy as np
import wandb
import h5py

# custom module
from brain_age_prediction import utils

##################################################################################
def load_matrix(sub_id, schaefer_variant, matrix_dir='/ritter/share/projects/laura_riedel_thesis/schaefer_fc_matrices.hdf5'):
    """
    Load a single subject-specific FC matrix from a HDF5 file.
    """
    with h5py.File(matrix_dir, 'r') as f:
        matrix = f['bids'][str(sub_id)][schaefer_variant+'_triu'][()]
    return matrix

def load_datasplit(split, schaefer_variant, 
                   datasplit_dir='../../data/schaefer/', 
                   matrix_dir='/ritter/share/projects/laura_riedel_thesis/schaefer_fc_matrices.hdf5'):
    """
    Loads specified data split of specified Schaefer variant from HDF5 file and 
    outputs a 3D matrix of the data (X) and the subjects' ages (y).
    Input:
        split: data split as string. Either 'train','val' or 'test'.
        schaefer_variant: which variant to load. Expects string of the form 'XnYp' where X is the number of
            networks and p is the number of parcellations. Possible values: '7n100p','7n200p','7n500p',
            '7n700p','7n1000p','17n100p','17n200p','17n500p','17n700p','17n1000p'. 
        datasplit_dir: path to where the 'data_info/' directory is located.
        matrix_dir: path to where the Schaefer FC matrices are stored. Expects a HDF5 file.
    Output:
        X: 3D array of stacked FC matrices relevant for the split.
        y: labels aka ages of subjects as 1D array
    """
    # get overview
    data_df = utils.get_data_overview_with_splitinfo(datasplit_dir)
    split_df = data_df.loc[data_df['split']==split]
    # derive information
    y = split_df['age'].to_numpy()
    sub_ids = split_df['eid'].to_numpy()
    # stack all subject FC matrices in one array
    X = np.stack([load_matrix(sub_id,schaefer_variant,matrix_dir) for sub_id in sub_ids])
    return X, y

def load_dataset(schaefer_variant, 
                 datasplit_dir='../../data/schaefer/',  
                 matrix_dir='/ritter/share/projects/laura_riedel_thesis/schaefer_fc_matrices.hdf5'):
    """
    Input:
        schaefer_variant: which variant to load. Expects string of the form 'XnYp' where X is the number of
            networks and p is the number of parcellations. Possible values: '7n100p','7n200p','7n500p',
            '7n700p','7n1000p','17n100p','17n200p','17n500p','17n700p','17n1000p'. 
        datasplit_dir: path to where the 'data_info/' directory is located.
        matrix_dir: path to where the Schaefer FC matrices are stored. Expects a HDF5 file.
    Output:
        For each split (train/val/test).
            X: 3D array of stacked FC matrices relevant for the split.
            y: labels aka ages of subjects as 1D array
    """
    print('Load train set...')
    X_train, y_train = load_datasplit('train', schaefer_variant, datasplit_dir, matrix_dir)
    print('Load val set...')
    X_val, y_val = load_datasplit('val', schaefer_variant, datasplit_dir, matrix_dir)
    print('Load test set...')
    X_test, y_test = load_datasplit('test', schaefer_variant, datasplit_dir, matrix_dir)
    return X_train, y_train, X_val, y_val, X_test, y_test

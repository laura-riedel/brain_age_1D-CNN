import numpy as np
import pandas as pd
import wandb
import h5py
from scipy.stats import zscore
from sklearn.decomposition import IncrementalPCA

# scikit-learn
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# custom module
from brain_age_prediction import utils

##################################################################################
# DATA ACCESSING / LOADING
def load_matrix(sub_id, schaefer_variant, remove_0=False, flatten=False,
                matrix_dir='/ritter/share/projects/laura_riedel_thesis/schaefer_fc_matrices.hdf5'):
    """
    Load a single subject-specific FC matrix from a HDF5 file.
    """
    with h5py.File(matrix_dir, 'r') as f:
        matrix = f['bids'][str(sub_id)][schaefer_variant+'_triu'][()]
    if remove_0:
        triu = np.triu_indices(matrix.shape[0], k=1)
        matrix = matrix[triu]
    else:
        if flatten:
            matrix = matrix.ravel()
    return matrix

def load_datasplit(split, schaefer_variant, remove_0=False, flatten=False, normalise=False,
                   datasplit_dir='../../data/schaefer/',
                   matrix_dir='/ritter/share/projects/laura_riedel_thesis/schaefer_fc_matrices.hdf5'):
    """
    Loads specified data split of specified Schaefer variant from HDF5 file and 
    outputs a 3D matrix of the data (X) and the subjects' ages (y).
    Input:
        split: data split as string. Either 'train','val','test' or 'heldout_test'.
        schaefer_variant: which variant to load. Expects string of the form 'XnYp' where X is the number of
            networks and p is the number of parcellations. Possible values: '7n100p','7n200p','7n500p',
            '7n700p','7n1000p','17n100p','17n200p','17n500p','17n700p','17n1000p'. 
        remove_0: Boolean flag whether to remove the 0s in the matrix; also flattens matrix to 1D array. 
            Default: False.
        flatten: Boolean flag whether to flatten the matrix (without removing 0s). Default: False.
        normalise: Boolean flag whether to normalise each subjects FC matrix. Default: False.
        datasplit_dir: path to where the 'data_info/' directory is located.
        matrix_dir: path to where the Schaefer FC matrices are stored. Expects a HDF5 file.
    Output:
        X: 3D array of stacked FC matrices relevant for the split.
        y: labels aka ages of subjects as 1D array
    """
    # get overview
    if split in ['train','val','test']:
        data_df = utils.get_data_overview_with_splitinfo(datasplit_dir)
        split_df = data_df.loc[data_df['split']==split]
    elif split == 'heldout_test':
        split_df = pd.read_csv(datasplit_dir)
    # derive information
    y = split_df['age'].to_numpy()
    sub_ids = split_df['eid'].to_numpy()
    # stack all subject FC matrices in one array
    X = np.stack([load_matrix(sub_id,schaefer_variant,remove_0,flatten,matrix_dir) for sub_id in sub_ids])
    if normalise:
        X = zscore(X, axis=0)
    return X, y

def load_dataset(schaefer_variant, remove_0=False, flatten=False, normalise=False,
                 datasplit_dir='../../data/schaefer/',  
                 matrix_dir='/ritter/share/projects/laura_riedel_thesis/schaefer_fc_matrices.hdf5'):
    """
    Loads train, val and test datasplits of specified Schaefer variant from HDF5 file.
    Input:
        schaefer_variant: which variant to load. Expects string of the form 'XnYp' where X is the number of
            networks and p is the number of parcellations. Possible values: '7n100p','7n200p','7n500p',
            '7n700p','7n1000p','17n100p','17n200p','17n500p','17n700p','17n1000p'. 
        remove_0: Boolean flag whether to remove the 0s in the matrix; also flattens matrix to 1D array. 
            Default: False.
        flatten: Boolean flag whether to flatten the matrix (without removing 0s). Default: False.
        normalise: Boolean flag whether to normalise each subjects FC matrix. Default: False.
        datasplit_dir: path to where the 'data_info/' directory is located.
        matrix_dir: path to where the Schaefer FC matrices are stored. Expects a HDF5 file.
    Output:
        For each split (train/val/test):
            X: 3D array of stacked FC matrices relevant for the split.
            y: labels aka ages of subjects as 1D array
    """
    print('Load train set...')
    X_train, y_train = load_datasplit('train', schaefer_variant, remove_0, flatten, normalise, datasplit_dir, matrix_dir)
    print('Load val set...')
    X_val, y_val = load_datasplit('val', schaefer_variant, remove_0, flatten, normalise, datasplit_dir, matrix_dir)
    print('Load test set...')
    X_test, y_test = load_datasplit('test', schaefer_variant, remove_0, flatten, normalise, datasplit_dir, matrix_dir)
    return X_train, y_train, X_val, y_val, X_test, y_test

def access_datasplit(split, schaefer_variant, no_0=False, normalise=False, shortcut='100-500p',
                     matrix_dir='/ritter/share/projects/laura_riedel_thesis/schaefer_fc_matrices.hdf5'):
    """
    Loads specified data split of specified Schaefer variant from the dataset shortcut
    in the HDF5 file and outputs a 3D or 2D matrix of the data (X) and the subjects' ages (y).
    Input:
        split: data split as string. Either 'train','val','test' or 'heldout_test'.
        schaefer_variant: which variant to load. Expects string of the form 'XnYp' where X is the number of
            networks and p is the number of parcellations. Possible values: '7n100p','7n200p','7n500p',
            '7n700p','7n1000p','17n100p','17n200p','17n500p','17n700p','17n1000p'. 
        no_0: Boolean flag whether to include the 0s in the matrix; also flattens matrix to 1D array. 
            Default: False.
        normalise: Boolean flag whether to normalise each subjects FC matrix. Default: False.
        shortcut: name of the shortcut directory to use; based on the underlying heldout test set used.
            Default: '100-500p'
        matrix_dir: path to where the Schaefer FC matrices are stored. Expects a HDF5 file.
    Output:
        X: 3D (or 2D, if flatten=True) array of stacked FC matrices relevant for the split.
        y: labels aka ages of subjects as 1D array
    """
    with h5py.File(matrix_dir, 'r') as f:
        addendum = ''
        if no_0:
            addendum = '_no-0'
        else:
            addendum = '_flattened'
        X = f['split_shortcuts'][shortcut][schaefer_variant][split]['X'+addendum][()]
        y = f['split_shortcuts'][shortcut][schaefer_variant][split]['y'][()]
        if normalise:
            X = zscore(X, axis=0)
    return X, y

def access_dataset(schaefer_variant, no_0=False, normalise=False, shortcut='100-500p',
                   matrix_dir='/ritter/share/projects/laura_riedel_thesis/schaefer_fc_matrices.hdf5'):
    """
    Loads train, val and test datasplits of specified Schaefer variant from 
    the dataset shortcut in the HDF5 file.
    Input:
        schaefer_variant: which variant to load. Expects string of the form 'XnYp' where X is the number of
            networks and p is the number of parcellations. Possible values: '7n100p','7n200p','7n500p',
            '7n700p','7n1000p','17n100p','17n200p','17n500p','17n700p','17n1000p'. 
        no_0: Boolean flag whether to include the 0s in the matrix; also flattens matrix to 1D array. 
            Default: False.
        normalise: Boolean flag whether to normalise each subjects FC matrix. Default: False.
        shortcut: name of the shortcut directory to use; based on the underlying heldout test set used.
            Default: '100-500p'.
        normalise: Boolean flag whether to normalise all X inputs. Default: False.
        matrix_dir: path to where the Schaefer FC matrices are stored. Expects a HDF5 file.
    Output:
        For each split (train/val/test):
            X: 3D (or 2D, if flatten=True) array of stacked FC matrices relevant for the split.
            y: labels aka ages of subjects as 1D array
    """
    X_train, y_train = access_datasplit('train', schaefer_variant, no_0, normalise, shortcut, matrix_dir)
    X_val, y_val = access_datasplit('val', schaefer_variant, no_0, normalise, shortcut, matrix_dir)
    X_test, y_test = access_datasplit('test', schaefer_variant, no_0, normalise, shortcut, matrix_dir)
    
    return X_train, y_train, X_val, y_val, X_test, y_test

##################################################################################
# PCA
def fit_transform_IPCA(X_train, X_val, X_test, n_components=500, incremental_batch_size=2500):
    """
    Takes the datasets of FC matrices, performs an incremental PCA on the training data
    and transforms the validation and test data.
    """
    utils.make_reproducible()

    ipca = IncrementalPCA(n_components=n_components, batch_size=incremental_batch_size)
    X_train_ipca = ipca.fit_transform(X_train)
    X_val_ipca = ipca.transform(X_val)
    X_test_ipca = ipca.transform(X_test)
    return X_train_ipca, X_val_ipca, X_test_ipca

##################################################################################
# W&B TRAINING / TESTING

def wandb_train_ridge(config, name=None, tags=None, schaefer_variant=None, shortcut=None,
                      alpha=None, no_0=None, normalise=None, ipca=False, n_components=500,
                      incremental_batch_size=2500, seed=43, plot=True):
    """
    Function for training a model with Schaefer FC matrices using external config
    information. Logs to W&B. Optional plot generation.
    """
    if name is None:
        name = config['name']
    if tags is None:
        tags = config['tags']

    # start wandb run
    with wandb.init(project=config['project'],
                    group=config['group'],
                    name=name,
                    tags=tags,
                    config=config['parameters']) as run:
        # update config with additional settings
        if schaefer_variant is not None:
            run.config['schaefer_variant'] = schaefer_variant
        if shortcut is not None:
            run.config['shortcut'] = shortcut
        if alpha is not None:
            run.config['alpha'] = alpha
        run.config['no_0'] = no_0
        run.config['normalise'] = normalise
        run.config['ipca'] = ipca
        if ipca:
            run.config['n_components'] = n_components
            run.config['incremental_batch_size'] = incremental_batch_size
        run.config['seed'] = seed
        updated_config = run.config
        
        # increase reproducibility
        utils.make_reproducible(updated_config.seed)
        
        # get data
        X_train, y_train, X_val, y_val, X_test, y_test = access_dataset(schaefer_variant=updated_config.schaefer_variant,
                                                                        no_0=updated_config.no_0,
                                                                        normalise=updated_config.normalise,
                                                                        shortcut=updated_config.shortcut,
                                                                        matrix_dir=updated_config.matrix_dir)
        if ipca:
            X_train, X_val, X_test = fit_transform_IPCA(X_train, X_val, X_test,
                                                        updated_config['n_components'],
                                                        updated_config['incremental_batch_size'])

        # define model
        ridge_model = Ridge(alpha=updated_config.alpha,
                            random_state=updated_config.seed)
        
        # train
        ridge_model.fit(X_train, y_train)
        
        # val
        y_pred_val = ridge_model.predict(X_val)
        run.summary['val_mae'] = mean_absolute_error(y_val, y_pred_val)
        run.summary['val_loss'] = mean_squared_error(y_val, y_pred_val)
        run.summary['best_val_loss'] = mean_squared_error(y_val, y_pred_val)
        run.summary['val_r2'] = r2_score(y_val, y_pred_val)
        
        # test
        y_pred_test = ridge_model.predict(X_test)
        run.summary['test_mae'] = mean_absolute_error(y_test, y_pred_test)
        run.summary['test_loss'] = mean_squared_error(y_test, y_pred_test)
        run.summary['test_r2'] = r2_score(y_test, y_pred_test)
        
        if plot:
            wandb.sklearn.plot_regressor(model=ridge_model,
                                         X_train=X_train,
                                         X_test=X_test,
                                         y_train=y_train,
                                         y_test=y_test,
                                         model_name=name)
        
        
    wandb.finish()
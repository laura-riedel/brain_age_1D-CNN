import numpy as np
import wandb
import h5py

# scikit-learn
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

# custom module
from brain_age_prediction import utils

##################################################################################
def load_matrix(sub_id, schaefer_variant, 
                matrix_dir='/ritter/share/projects/laura_riedel_thesis/schaefer_fc_matrices.hdf5'):
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
    Loads train, val and test datasplits of specified Schaefer variant from HDF5 file.
    Input:
        schaefer_variant: which variant to load. Expects string of the form 'XnYp' where X is the number of
            networks and p is the number of parcellations. Possible values: '7n100p','7n200p','7n500p',
            '7n700p','7n1000p','17n100p','17n200p','17n500p','17n700p','17n1000p'. 
        datasplit_dir: path to where the 'data_info/' directory is located.
        matrix_dir: path to where the Schaefer FC matrices are stored. Expects a HDF5 file.
    Output:
        For each split (train/val/test):
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

def access_datasplit(split, schaefer_variant, flatten=False, shortcut='100-500p',
                     matrix_dir='/ritter/share/projects/laura_riedel_thesis/schaefer_fc_matrices.hdf5'):
    """
    Loads specified data split of specified Schaefer variant from the dataset shortcut
    in the HDF5 file and outputs a 3D or 2D matrix of the data (X) and the subjects' ages (y).
    Input:
        split: data split as string. Either 'train','val' or 'test'.
        schaefer_variant: which variant to load. Expects string of the form 'XnYp' where X is the number of
            networks and p is the number of parcellations. Possible values: '7n100p','7n200p','7n500p',
            '7n700p','7n1000p','17n100p','17n200p','17n500p','17n700p','17n1000p'. 
        flatten: Boolean flag whether to transform the 3D data (X) into a 2D array. Default: False.
        shortcut: name of the shortcut directory to use; based on the underlying heldout test set used.
            Default: '100-500p'
        matrix_dir: path to where the Schaefer FC matrices are stored. Expects a HDF5 file.
    Output:
        X: 3D (or 2D, if flatten=True) array of stacked FC matrices relevant for the split.
        y: labels aka ages of subjects as 1D array
    """
    with h5py.File(matrix_dir, 'r') as f:
        X = f['split_shortcuts'][shortcut][schaefer_variant][split]['X'][()]
        y = f['split_shortcuts'][shortcut][schaefer_variant][split]['y'][()]
    if flatten:
        nsamples, nx, ny = X.shape
        X = X.reshape((nsamples,nx*ny))
    return X, y

def access_dataset(schaefer_variant, flatten=False, shortcut='100-500p',
                   matrix_dir='/ritter/share/projects/laura_riedel_thesis/schaefer_fc_matrices.hdf5'):
    """
    Loads train, val and test datasplits of specified Schaefer variant from 
    the dataset shortcut in the HDF5 file.
    Input:
        schaefer_variant: which variant to load. Expects string of the form 'XnYp' where X is the number of
            networks and p is the number of parcellations. Possible values: '7n100p','7n200p','7n500p',
            '7n700p','7n1000p','17n100p','17n200p','17n500p','17n700p','17n1000p'. 
        flatten: Boolean flag whether to transform the 3D data (X) into a 2D array. Default: False.
        shortcut: name of the shortcut directory to use; based on the underlying heldout test set used.
            Default: '100-500p'
        matrix_dir: path to where the Schaefer FC matrices are stored. Expects a HDF5 file.
    Output:
        For each split (train/val/test):
            X: 3D (or 2D, if flatten=True) array of stacked FC matrices relevant for the split.
            y: labels aka ages of subjects as 1D array
    """
    X_train, y_train = access_datasplit('train', schaefer_variant, flatten, shortcut, matrix_dir)
    X_val, y_val = access_datasplit('val', schaefer_variant, flatten, shortcut, matrix_dir)
    X_test, y_test = access_datasplit('test', schaefer_variant, flatten, shortcut, matrix_dir)
    return X_train, y_train, X_val, y_val, X_test, y_test

##################################################################################
# W&B training / testing

def wandb_train_ridge(config, name=None, tags=None, schaefer_variant=None, shortcut=None, alpha=None, seed=43, plot=True):
    """
    """
    if name is None:
        name = config['name']
    if tags is None:
        tags = config['tags']
        
        
    # start wandb run
    with wandb.init(project='lightweight-brain-age-prediction',
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
        run.config['seed'] = seed
        updated_config = run.config
        
        # increase reproducibility
        utils.make_reproducible(updated_config.seed)
        
        # get data
        X_train, y_train, X_val, y_val, X_test, y_test = access_dataset(schaefer_variant=updated_config.schaefer_variant,
                                                                        flatten=updated_config.flatten,                                           
                                                                        shortcut=updated_config.shortcut,
                                                                        matrix_dir=updated_config.matrix_dir)
        
        # define model
        ridge_model = Ridge(alpha=updated_config.alpha, 
                            random_state=updated_config.seed)
        
        # train
        ridge_model.fit(X_train, y_train)
        
        # test
        y_pred = ridge_model.predict(X_test)
        
        run.summary['test_mae'] = mean_absolute_error(y_test, y_pred)
        run.summary['test_loss'] = mean_squared_error(y_test, y_pred)
        
        
        if plot:
            wandb.sklearn.plot_regressor(model=ridge_model, 
                                         X_train=X_train, 
                                         X_test=X_test, 
                                         y_train=y_train, 
                                         y_test=y_test, 
                                         model_name=name)
        
        
    wandb.finish()
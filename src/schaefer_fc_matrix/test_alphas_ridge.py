"""
Script for testing different alpha values for a Ridge Regression model
trained on FC matrices to predict age.
Uses Weights & Biases for tracking.
"""
import numpy as np

# custom module
from brain_age_prediction import sklearn_utils

##################################################################################
# config
config = {'project': 'lightweight-brain-age-prediction',
          'group': 'schaefer_fc_matrix',
          'name': '7n100p_ridge_alpha',
          'tags': ['ridge','7n','100p', 'tryout'],
          'parameters': {
              # dataset set-up
              'schaefer_variant': '7n100p',
              'shortcut': '100-500p',
              'shared_variants': ['7n100p','7n200p','7n500p','17n100p','17n200p','17n500p'],
              'matrix_dir': '/ritter/share/projects/laura_riedel_thesis/schaefer_fc_matrices.hdf5'},
         }

# train
for alpha in np.logspace(-5,5,20):
    print(f'>> {alpha}')
    updated_tags = config['tags']+[f'alpha {alpha}']
    updated_name = config['name']+'-'+str(alpha)
    print('- no 0, normalised')
    sklearn_utils.wandb_train_ridge(config, 
                                    name=updated_name, 
                                    tags=updated_tags+['no 0', 'normalised'],
                                    alpha=alpha,  
                                    no_0=True,
                                    normalise=True,
                                    plot=False)
    print('- no 0, not normalised')
    sklearn_utils.wandb_train_ridge(config, 
                                    name=updated_name, 
                                    tags=updated_tags+['no 0'],
                                    alpha=alpha,  
                                    no_0=True,
                                    normalise=False,
                                    plot=False)
    print('- with 0, normalised')
    sklearn_utils.wandb_train_ridge(config, 
                                    name=updated_name, 
                                    tags=updated_tags+['with 0', 'normalised'],
                                    alpha=alpha,  
                                    no_0=False,
                                    normalise=True,
                                    plot=False)
    print('- with 0, not normalised')
    sklearn_utils.wandb_train_ridge(config, 
                                    name=updated_name, 
                                    tags=updated_tags+['with 0'],
                                    alpha=alpha,  
                                    no_0=False,
                                    normalise=False,
                                    plot=False)
    print('\n')
print('DONE.')
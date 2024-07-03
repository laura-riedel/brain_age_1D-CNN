"""
Script for training a Ridge Regression model on Schaefer FC matrices to predict age.
All variables are determined through the config yaml the script is called with.
Uses Weights & Biases for tracking.
"""

import yaml
import argparse

# custom module
from brain_age_prediction import sklearn_utils

##################################################################################
parser = argparse.ArgumentParser(prog='RunRidgeFC',
                                description='Trains and tests a model configuration with a Ridge Regression on UKBB FC data.',
                                epilog='Training + testing complete.')

parser.add_argument('config_path', type=str,
                    help='Enter the path to the config yaml to be used.')
# input for wandb_train
parser.add_argument('--name', default=None,
                    help='enter name for the W&B run. Defaults to the name in the config.')
parser.add_argument('--tags', default=None,
                    help='Enter tags for the W&B run. Defaults to the tags in the config.')
parser.add_argument('--schaefer_variant', default=None,
                    help='Enter the name of the Schaefer variant to use. Defaults to the variant in the config.')
parser.add_argument('--shortcut', default=None,
                    help='Enter the data shortcut to use. Defaults to the shortcut in the config.')
parser.add_argument('--alpha', default=None,
                    help='Enter the alpha value to use in the Ridge Regression. Defaults to the alpha in the config.')
parser.add_argument('--no_0', type=bool, default=True,
                    help='Enter boolean flag whether to exclude the zeros from the matrices. Default: True.')
parser.add_argument('--normalise', type=bool, default=True,
                    help='Enter boolean flag whether to normalise input before training. Default: True.')
parser.add_argument('--ipca', type=bool, default=False,
                    help='Enter boolean flag whether to perform incremental PCA. Default: False.')
parser.add_argument('--n_components', type=int, default=500,
                    help='Enter number of components to be extracted during PCA. Default: 500.')
parser.add_argument('--incremental_batch_size', type=int, default=2500,
                    help='Enter size of minibatches for incremental PCA. Default: 2500.')
parser.add_argument('--seed', type=int, default=43,
                    help='Enter random seed to be used. Default: 43.')
parser.add_argument('--plot', type=bool, default=False,
                    help='Enter boolean flag whether to create regression plots. Default: False.')

args = parser.parse_args()
##################################################################################
# load config
with open(args.config_path, 'r') as f:
    config = yaml.safe_load(f)

# train
sklearn_utils.wandb_train_ridge(config,
                                name=args.name,
                                tags=args.tags,
                                schaefer_variant=args.schaefer_variant,
                                shortcut=args.shortcut,
                                alpha=args.alpha,
                                no_0=args.no_0,
                                normalise=args.normalise,
                                ipca=args.ipca,
                                n_components=args.n_components,
                                incremental_batch_size=args.incremental_batch_size,
                                seed=args.seed,
                                plot=args.plot)
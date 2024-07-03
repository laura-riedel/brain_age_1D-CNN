"""
Script for training a lightweight variableCNN model on Schaefer timeseries to predict age.
All variables are determined through the config yaml the script is called with.
Uses Weights & Biases for tracking.
"""
import argparse
import yaml

# custom module
from brain_age_prediction import wandb_utils

##################################################################################
parser = argparse.ArgumentParser(prog='RunVariableCNN',
                                description='Trains and tests a model configuration with variableCNN on UKBB data.',
                                epilog='Training + testing complete.')

parser.add_argument('config_path', type=str,
                    help='Enter the path to the config yaml to be used.')
# input for wandb_train
parser.add_argument('--name', default=None,
                    help='enter name for the W&B run. Defaults to the name in the config.')
parser.add_argument('--tags', default=None,
                    help='Enter tags for the W&B run. Defaults to the tags in the config.')
parser.add_argument('--use_gpu', type=bool, default=False,
                    help='Enter boolean flag to indicate which accelerator to use for training. Uses GPU if True and CPU if False. Default: False.')
parser.add_argument('--devices', default=None,
                    help='Enter the devices to use when training; depends on accelerator. Default for GPU use: [1].')
parser.add_argument('--dev', type=bool, default=True,
                    help='Enter boolean flag to indicate whether model training/testing is still in the development phase. If True, held-out IDs are dropped from meta_df, if False, only held-out IDs are kept. Default: True.')
parser.add_argument('--batch_size', type=int, default=128,
                    help='Enter batch size for DataLoaders. Default: 128.')
parser.add_argument('--max_epochs', default=None,
                    help='Enter for how many epochs the model is supposed to train. Defaults to max_epochs in the config.')
parser.add_argument('--num_threads', type=int, default=2,
                    help='If use_gpu=False, enter how many CPU threads to use. Default: 2.')
parser.add_argument('--seed', type=int, default=43,
                    help='Enter random seed to be used. Default: 43.')
parser.add_argument('-lr_config_path','--lr_scheduler_config_path', default=None,
                    help='Enter the path to the learning rate scheduler config, if applicable. Default: None.')
parser.add_argument('--train_ratio', type=float, default=0.88,
                    help='Enter first parameter for train/val/test split regulation. On a scale from 0 to 1, which proportion of data is to be used for training? Default: 0.88.')
parser.add_argument('--val_test_ratio', type=bool, default=0.5,
                    help='Enter second parameter for train/val/test split regulation. On a sclae form 0 to 1, which proportion of the split not used for training is to be used for validating/testing? >0.5: more data for validation; <0.5: more data for testing. Default: 0.5.')
parser.add_argument('--save_datasplit', type=float, default=True,
                    help='Enter boolean flag whether to save the applied data split as W&B artifact. Default: True.')
parser.add_argument('--save_overview', type=bool, default=False,
                    help='Enter boolean flag whether to save the idx/age overview as W&B artifact. Only effective with save_datasplit=True. Default: False.')
parser.add_argument('--all_data', type=bool, default=True,
                    help='Enter boolean flag to indicate whether to use all data or only a subset of 100 samples. Default: True.')
parser.add_argument('--test', type=bool, default=True,
                    help='Enter boolean flag whether to test the trained model\'s performance on the test set. Default: True.')
parser.add_argument('--finish', type=bool, default=True,
                    help='Enter boolean flag whether to finish a wandb run. Default: True.')

args = parser.parse_args()
##################################################################################
# load config
with open(args.config_path, 'r') as f:
    config = yaml.safe_load(f)
# with open(args.lr_scheduler_config_path, 'r') as f:
#     lr_scheduler_config = yaml.safe_load(f)

# train
wandb_utils.wandb_train(config,
                        name=args.name,
                        tags=args.tags,
                        use_gpu=args.use_gpu,
                        devices=args.devices,
                        dev=args.dev,
                        batch_size=args.batch_size,
                        max_epochs=args.max_epochs,
                        num_threads=args.num_threads,
                        seed=args.seed,
                        lr_scheduler_config_path=args.lr_scheduler_config_path,
                        train_ratio=args.train_ratio,
                        val_test_ratio=args.val_test_ratio,
                        save_datasplit=args.save_datasplit,
                        save_overview=args.save_overview,
                        all_data=args.all_data,
                        test=args.test,
                        finish=args.finish,
                        execution='t')
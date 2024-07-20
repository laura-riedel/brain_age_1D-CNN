"""
Script for testing different seeds for a 1D-CNN model
trained on Schaefer timeseries to predict age.
Uses Weights & Biases for tracking.
"""
import yaml
# custom module
from brain_age_prediction import utils, wandb_utils

##################################################################################
seeds = [0,10,27,43,56,87,135,274,583,999]
config_path = 'parameters/ConstantLR/double_conv-True/7n100p_original_conv-scale-False.yaml'

# load config
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# change config group
config['group'] = 'schaefer_ts_test_certainty_range'

for seed in seeds:
    print(f'>> {seed}')
    utils.seed_everything(seed)

    # make name + tags run specific
    run_name = config['name']+'_seed-'+str(seed)
    run_tags = config['tags']+[f'seed {seed}']

    # train
    wandb_utils.wandb_train(config,
                            name=run_name,
                            tags=run_tags,
                            use_gpu=False,
                            dev=True,
                            batch_size=128,
                            num_threads=2,
                            seed=seed,
                            train_ratio=0.88,
                            val_test_ratio=0.5,
                            save_datasplit=True,
                            save_overview=True,
                            all_data=True,
                            test=True,
                            finish=True,
                            execution='t')
    print('\n')
print('DONE.')

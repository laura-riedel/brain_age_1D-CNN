import pandas as pd
import numpy as np
import os
import re
import random
import logging
import yaml
from typing import Any, Dict, Generator, Optional
from sklearn.linear_model import LinearRegression

# Pytorch etc
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

# import data + model modules
from brain_age_prediction import data
from brain_age_prediction import models

# visualisation
import matplotlib.pyplot as plt
import seaborn as sns

##################################################################################
### SAVING THINGS
def save_data_info(path, datamodule, only_indices=False):
    """
    Creates a new directory "data_info" and saves the loaded overview of used datapoints
    (participant ID, age, file location) and the indices used for the train, validation,
    and test set in separate files in that new directory.
    """
    # create data_info directory
    os.makedirs(path+'data_info', exist_ok=True)
    if not only_indices:
        # save overview dataframe as csv additional to indices
        datamodule.data.labels.to_csv(path+'data_info/overview.csv', index=False)
    # save train/val/test indices as integers ('%i') in csv
    np.savetxt(path+'data_info/train_idx.csv', datamodule.train_idx, fmt='%i')
    np.savetxt(path+'data_info/val_idx.csv', datamodule.val_idx, fmt='%i')
    np.savetxt(path+'data_info/test_idx.csv', datamodule.test_idx, fmt='%i')

### REPRODUCIBILITY
# increase reproducitility by defining a seed worker
# instructions from https://pytorch.org/docs/stable/notes/randomness.html#reproducibility
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    
# define a function for easier set-up
def make_reproducible(random_state=43):
    """
    Function that calls on 'seed_everything' to set seeds for pseudo-random number generators
    and that enforeces more deterministic behaviour in torch.
    Input:
        random_state: seed to set
    """
    seed_everything(seed=random_state, workers=True)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed(random_state)

# copy seed_everything function provided in Lightning vers. 1.6.3 that is not 
# available anymore in 2.0.2 from 
# https://pytorch-lightning.readthedocs.io/en/1.6.3/_modules/pytorch_lightning/utilities/seed.html#seed_everything
log = logging.getLogger(__name__)
def seed_everything(seed: Optional[int] = None, workers: bool = False) -> int:
    """Function that sets seed for pseudo-random number generators in: pytorch, numpy, python.random In addition,
    sets the following environment variables:

    - `PL_GLOBAL_SEED`: will be passed to spawned subprocesses (e.g. ddp_spawn backend).
    - `PL_SEED_WORKERS`: (optional) is set to 1 if ``workers=True``.

    Args:
        seed: the integer value seed for global random state in Lightning.
            If `None`, will read seed from `PL_GLOBAL_SEED` env variable
            or select it randomly.
        workers: if set to ``True``, will properly configure all dataloaders passed to the
            Trainer with a ``worker_init_fn``. If the user already provides such a function
            for their dataloaders, setting this argument will have no influence. See also:
            :func:`~pytorch_lightning.utilities.seed.pl_worker_init_function`.
    """
    max_seed_value = np.iinfo(np.uint32).max
    min_seed_value = np.iinfo(np.uint32).min

    if seed is None:
        env_seed = os.environ.get("PL_GLOBAL_SEED")
        if env_seed is None:
            seed = _select_seed_randomly(min_seed_value, max_seed_value)
            rank_zero_warn(f"No seed found, seed set to {seed}")
        else:
            try:
                seed = int(env_seed)
            except ValueError:
                seed = _select_seed_randomly(min_seed_value, max_seed_value)
                rank_zero_warn(f"Invalid seed found: {repr(env_seed)}, seed set to {seed}")
    elif not isinstance(seed, int):
        seed = int(seed)

    if not (min_seed_value <= seed <= max_seed_value):
        rank_zero_warn(f"{seed} is not in bounds, numpy accepts from {min_seed_value} to {max_seed_value}")
        seed = _select_seed_randomly(min_seed_value, max_seed_value)

    # using `log.info` instead of `rank_zero_info`,
    # so users can verify the seed is properly set in distributed training.
    log.info(f"Global seed set to {seed}")
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ["PL_SEED_WORKERS"] = f"{int(workers)}"

    return seed

# call reproducibility function
make_reproducible()

#### MODEL INITIALISATION
# logger
def logger_init(save_dir):
    logger = CSVLogger(save_dir=save_dir, name='Logs')
    return logger

# callbacks
def checkpoint_init(save_dir):
    checkpoint = ModelCheckpoint(dirpath=save_dir+'Checkpoint/',
                                 filename='models-{epoch:02d}-{val_loss:.2f}',
                                 monitor='val_loss',
                                 save_top_k=1,
                                 mode='min')
    return checkpoint

def earlystopping_init(patience=15):
    early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=patience)
    return early_stopping

# initialise trainer
def trainer_init(device, logger, log_steps=10, max_epochs=175, callbacks=[]):
    trainer = pl.Trainer(accelerator='gpu',
                         devices=[device], 
                         logger=logger,
                         log_every_n_steps=log_steps,
                         max_epochs=max_epochs,
                         callbacks=callbacks,
                         deterministic=True)
    return trainer

#### TRAIN + TEST MODELS WITH CONFIG DICT; PREDICT AGE WITH TRAINED MODEL
def train_model(log_path, data_path, config, device, execution='nb'):
    """
    Fuction for training a variable ICA 1D-CNN model on a GPU using external config information.
    Outputs a trained model.
    Input:
        log_path: path to where logs, checkpoints and data info should be saved
        data_path: path to location where data is saved (expectations see ukbb_data.DataModule)
        config: configuration dictionary of the form 
                {'project': '', 'model': '', 'parameters': {all parameters except for execution}}
        device: which GPU to run on
        execution: whether model is called from a Jupyter Notebook ('nb') or the terminal ('t'). 
                    Teriminal call cannot handle dilation. Default: 'nb'.
    Output:
        trainer: the trainer instance of the model 
        variable_CNN: the trained model
        datamodule: PyTorch Lightning UKBB DataModule
    """
    full_log_path = log_path+config['project']+'/'+config['model']+'/'
    
    # initialise model
    variable_CNN = ukbb_ica_models.variable1DCNN(in_channels=config['parameters']['in_channels'],
                                                kernel_size=config['parameters']['kernel_size'],
                                                lr=config['parameters']['lr'],
                                                depth=config['parameters']['depth'],
                                                start_out=config['parameters']['start_out'],
                                                stride=config['parameters']['stride'],
                                                conv_dropout=config['parameters']['conv_dropout'],
                                                final_dropout=config['parameters']['final_dropout'],
                                                weight_decay=config['parameters']['weight_decay'],
                                                dilation=config['parameters']['dilation'],
                                                double_conv=config['parameters']['double_conv'],
                                                batch_norm=config['parameters']['batch_norm'],
                                                execution=execution)

    # initialise logger
    logger = logger_init(save_dir=full_log_path)

    # set callbacks
    early_stopping = earlystopping_init(patience=config['parameters']['patience'])

    checkpoint = checkpoint_init(save_dir=full_log_path)

    # initialise trainer
    trainer = trainer_init(device=device,
                                logger=logger,
                                log_steps=config['parameters']['log_steps'],
                                max_epochs=config['parameters']['max_epochs'],
                                callbacks=[early_stopping, checkpoint])

    # initialise DataModule
    datamodule = ukbb_data.UKBBDataModule(data_path,
                                         ica=config['parameters']['ica'],
                                         good_components=config['parameters']['good_components'])

    # train model
    trainer.fit(variable_CNN, datamodule=datamodule)
    print('Training complete.')

    # save info on which data was used + what the train/val/test split was
    save_data_info(path=full_log_path, datamodule=datamodule)
    
    return trainer, variable_CNN, datamodule

def predict_w_model(trainer, model, datamodule, log_path):
    """
    Function for predicting the brain age of subjects in the validation + test set
    using a previously trained model. Automatically saves predictions in data_info.
    Input:
        trainer: the trainer instance of the model model
        model: the trained model
        datamodule: PyTorch Lightning UKBB DataModule instance
        log_path: path to where logs, checkpoints and data info are saved
    Output:
        preds_df: dataframe with predicted ages
    """
    model.eval()
    predictions = trainer.predict(model, datamodule)
    preds_df = pd.DataFrame(columns=['eid','batch_nb','predicted_age'])
    count = 0
    for batch_number, batch in enumerate(predictions):
        for i in range(len(batch[0])):
            preds_df.loc[count,'eid'] = int(batch[0][i])
            preds_df.loc[count,'batch_nb'] = batch_number
            preds_df.loc[count,'predicted_age'] = float(batch[1][i])       
            count += 1
    
    preds_df.to_csv(log_path+'data_info/predictions.csv', index=False)
    return preds_df
    

def test_model(trainer, datamodule, config):
    """
    Fuction for using the same model and testing set-up using external config information.
    Outputs a test score and a plot visualising the training progression.
    Input:
        trainer: the trainer instance of the model model
        datamodule: PyTorch Lightning UKBB DataModule instance
        config: configuration dictionary of the form 
                {'project': '', 'model': '', 'parameters': {all parameters except for execution}}
    """
    model_info = config['model']
    ica_info = config['parameters']['ica']
    if config['parameters']['good_components'] == True:
        gc = 'good components only'
    else:
        gc = 'all components'
    if config['project'] == 'FinalModels_IM':
        data_info = '(data subset)'
    else:
        data_info = '(full data)'
    
    # test model
    print(f'\nTesting model "{model_info}" {data_info}...')
    trainer.test(ckpt_path='best', datamodule=datamodule)
    
    # visualise training
    print(f'\nVisualise training of model "{model_info}" {data_info}...')    
    metrics = get_current_metrics(trainer, show=True)
    plot_training(data=metrics, title=f'Training visualisation of the ICA{ica_info} 1D-CNN with {gc} {data_info}.')

#### LOADING DATA
def get_current_metrics(trainer, show=False):
    """
    Gets the metrics of a current trainer object (aka the training + validation information
    of the trained with the trainer) and returns them in a dataframe.
    """
    metrics = pd.read_csv(f'{trainer.logger.log_dir}/metrics.csv')
    # set epoch as index
    metrics.set_index('epoch', inplace=True)
    # move step to first position
    step = metrics.pop('step')
    metrics.insert(0, 'step', step)
    # display first 5 rows if required
    if show:
        display(metrics.dropna(axis=1, how='all').head())
    return metrics

def get_saved_metrics(logging_path, show=False):
    """
    Gets the metrics of saved training + validation information
    and returns them in a dataframe.
    """
    metrics = pd.read_csv(f'{logging_path}metrics.csv')
    # set epoch as index
    metrics.set_index('epoch', inplace=True)
    # move step to first position
    step = metrics.pop('step')
    metrics.insert(0, 'step', step)
    # display first 5 rows if required
    if show:
        display(metrics.dropna(axis=1, how='all').head())
    return metrics

def load_datainfo(path_to_data_info, info_type):
    """
    Loads information from the data_info directory.
    Input:
        path_to_data_info: path to where data_info is saved (without data_info itself in path)
        info_type: 'train_idx', 'val_idx', 'test_idx', or 'overview'
    Output:
        info: if info_type is one of the split indices: numpy array of the split indices
              if info_type is overview: overview.csv as dataframe  
    """
    if info_type.endswith('idx'):
        info = np.loadtxt(path_to_data_info+'data_info/'+info_type+'.csv', dtype='int')
    else:
        info = pd.read_csv(path_to_data_info+'data_info/overview.csv')
    return info

def get_data_overview_with_splitinfo(path_to_data_info, index=False):
    """
    Loads information from the data_info directory about participants' ages 
    and in which split participants were present during model training/testing.
    Input:
        path_to_data_info: path to where data_info is saved (without data_info itself in path)
        index: whether an item is retrieved based on index (=True) or based on subject id (=False). Default: False.
    Output:
        data_overview: participant/age/split overview as dataframe  
    """
    data_overview = load_datainfo(path_to_data_info, 'overview')
    train_idx = load_datainfo(path_to_data_info, 'train_idx')
    val_idx = load_datainfo(path_to_data_info, 'val_idx')
    test_idx = load_datainfo(path_to_data_info, 'test_idx')
    
    # add split information to items in data_overview
    if index:
        # based on index
        for idx in train_idx:
            data_overview.loc[idx, 'split'] = 'train'
        for idx in val_idx:
            data_overview.loc[idx, 'split'] = 'val'
        for idx in test_idx:
            data_overview.loc[idx, 'split'] = 'test'
    else:
        # based on sub ID
        for idx in train_idx:
            data_overview.loc[data_overview['eid'] == idx, 'split'] = 'train'
        for idx in val_idx:
            data_overview.loc[data_overview['eid'] == idx, 'split'] = 'val'
        for idx in test_idx:
            data_overview.loc[data_overview['eid'] == idx, 'split'] = 'test'
            
    return data_overview

def get_metadata(path_to_data_info, ukbb_data_path, index=True):
    """
    Get additional information about participants used in model training/testing; 
    also add in which split a participant was present.
    Input:
        path_to_data_info: path to where data_info is saved (without data_info itself in path)
        ukbb_data_path: path to ukb_data directory (ukb_data included). E.g. 'ritter/share/data/UKBB/ukb_data'.
        index: whether an item is retrieved based on index (=True) or based on subject id (=False). Default: True.
    Output:
        meta_df: overview dataframe containing the IDs of included participants
                with additional information regarding which split they were in + additional metadata
    """
    data_overview = get_data_overview_with_splitinfo(path_to_data_info, index)
    
    # META INFORMATION
    # match additional info with subject IDs
    # get subject IDs, rename column for later merge
    ids_df = pd.read_csv(ukbb_data_path+'table/ukb_imaging_filtered_eids.txt')
    ids_df.rename(columns={'f.eid': 'eid'}, inplace=True)
    # get bmi 
    bmi_df = pd.read_csv(ukbb_data_path+'table/targets/bmi.tsv', sep='\t', names=['bmi'])
    # get digit substitution 
    digit_df = pd.read_csv(ukbb_data_path+'table/targets/digit-substitution.tsv', sep='\t', names=['digit substitution'])
    # get education 
    education_df = pd.read_csv(ukbb_data_path+'table/targets/edu.tsv', sep='\t', names=['education'])
    # get fluid intelligence 
    fluid_int_df = pd.read_csv(ukbb_data_path+'table/targets/fluid-intelligence.tsv', sep='\t', names=['fluid intelligence'])
    # get grip 
    grip_df = pd.read_csv(ukbb_data_path+'table/targets/grip.tsv', sep='\t', names=['grip'])
    # get depressive episode
    depressive_ep_df = pd.read_csv(ukbb_data_path+'table/targets/icd-f32-depressive-episode.tsv', sep='\t', names=['depressive episode'])
    # get depressive episode
    depression_all_df = pd.read_csv(ukbb_data_path+'table/targets/icd-f32-f33-all-depression.tsv', sep='\t', names=['all depression'])
    # get recurrent depressive order
    depressive_rec_df = pd.read_csv(ukbb_data_path+'table/targets/icd-f33-recurrent-depressive-disorder.tsv', sep='\t', names=['recurrent depressive disorder'])
    # get multiple sclerosis
    ms_df = pd.read_csv(ukbb_data_path+'table/targets/icd-g35-multiple-sclerosis.tsv', sep='\t', names=['multiple sclerosis'])
    # get sex 
    sex_df = pd.read_csv(ukbb_data_path+'table/targets/sex.tsv', sep='\t', names=['sex'])
    # get weekly beer
    weekly_beer_df = pd.read_csv(ukbb_data_path+'table/targets/weekly-beer.tsv', sep='\t', names=['weekly beer'])
    # get ethnicity
    ethnicity_df = pd.read_csv(ukbb_data_path+'table/features/genetic-pcs.tsv', sep='\t', usecols=[0,1,2], names=['genetic pc 1', 'genetic pc 2', 'genetic pc 3'])
    # combine additional information
    # important to combine BEFORE merging with data_overview in order to keep the correct ID mapping
    variables = [ids_df, bmi_df, digit_df, education_df, fluid_int_df, grip_df, depressive_ep_df, depression_all_df, depressive_rec_df, ms_df, sex_df, weekly_beer_df, ethnicity_df]
    meta_df = pd.concat(variables, axis=1)
    # merge with data_overview
    meta_df = data_overview.merge(meta_df, on='eid', how='left')
    return meta_df

def get_schaefer_overview(schaefer_data_dir='../../data/schaefer/', 
                          variants=['7n100p','7n200p','7n500p','7n700p','7n1000p','17n100p','17n200p','17n500p','17n700p','17n1000p']):
    """
    Collect all information about the availability of subject data in one dataframe.
    Input:
        schaefer_data_dir: path to local schaefer directory containing subfolders for each variant. Default: '../../data/schaefer/'.
        variants: list of variants of interest. Default: all variants.
    Output:
        schaefer_exists_df: overview pandas dataframe.
    """
    # initial df with first variant 
    schaefer_exists_df = pd.read_csv(schaefer_data_dir+variants[0]+'/schaefer_exists.csv')
    # add other variants if applicable
    if len(variants) > 1:
        for variant in variants[1:]:
            variant_df = pd.read_csv(schaefer_data_dir+variant+'/schaefer_exists.csv')
            schaefer_exists_df = schaefer_exists_df.merge(variant_df, how='left', on='eid', suffixes=(None,'_'+variant))
    # add first variant name to according columns 
    rename_dict = {'schaefer_exists': 'schaefer_exists_'+variants[0],
                   'is_empty': 'is_empty_'+variants[0],
                   'contains_nan': 'contains_nan_'+variants[0],
                   'contains_0': 'contains_0_'+variants[0],
                   'location_of_0': 'location_of_0_'+variants[0]}
    schaefer_exists_df.rename(columns=rename_dict, inplace=True)
    return schaefer_exists_df

def get_usable_schaefer_ids(schaefer_data_dir='../../data/schaefer/', 
                            variants=['7n100p','7n200p','7n500p','7n700p','7n1000p','17n100p','17n200p','17n500p','17n700p','17n1000p']):
    """
    Get those subject IDs for which we have usable Schaefer variant files.
    Input:
        schaefer_data_dir: path to local schaefer directory containing subfolders for each variant. Default: '../../data/schaefer/'.
        variants: list of variants of interest. Default: all variants.
    Output:
        list of relevant subject IDs
    """
    schaefer_exists_df = get_schaefer_overview(schaefer_data_dir, variants)
    for variant in variants:  
        schaefer_exists_df.drop(schaefer_exists_df[schaefer_exists_df['schaefer_exists_'+variant]==False].index, inplace=True)
        schaefer_exists_df.drop(schaefer_exists_df[schaefer_exists_df['is_empty_'+variant]==True].index, inplace=True)
        schaefer_exists_df.drop(schaefer_exists_df[schaefer_exists_df['contains_nan_'+variant]==True].index, inplace=True)
        schaefer_exists_df.drop(schaefer_exists_df[schaefer_exists_df['contains_0_'+variant]==True].index, inplace=True)
    return list(schaefer_exists_df['eid']) 

def get_heldout_schaefer_overview(ukbb_path='/ritter/share/data/UKBB/ukb_data/', 
                                  schaefer_data_dir='../../data/schaefer/',
                                  heldout_set_name='heldout_test_set_100-500p.csv'):
    """
    Creates an overview of IDs / ages for a specified heldout test set.
    """
    heldout_ids = set(np.loadtxt(schaefer_data_dir+heldout_set_name, dtype=int))
    # get target information (age)
    age_df = pd.read_csv(ukbb_path+'table/targets/age.tsv', sep='\t', names=['age'])
    # get subject IDs, rename column 
    ids_df = pd.read_csv(ukbb_path+'table/ukb_imaging_filtered_eids.txt')
    ids_df.rename(columns={'f.eid': 'eid'}, inplace=True)
    # combine information
    meta_df = pd.concat([ids_df, age_df], axis=1)
    # keep only held-out IDs
    meta_df = meta_df[meta_df['eid'].isin(heldout_ids)]
    # reset index
    meta_df.reset_index(drop=True, inplace=True)
    return meta_df

#### OTHER HELPER FUNCTIONS
def calculate_bag(df, single=False):
    """
    Calculate the brain age gap (BAG) for each participant in the dataframe.
    Input:
        df: dataframe including true and predicted ages
        single: Boolean indicating whether BAGs should be calculated for
                a single (=TRUE) ICA modality or both (=FALSE).
    Output:
        df: dataframe with additional BAG column(s)
    """
    if single:
        df['bag'] = df['predicted_age'] - df['age']
    else:
        df['bag_ICA25'] = df['predicted_age_ICA25'] - df['age']
        df['bag_ICA100'] = df['predicted_age_ICA100'] - df['age']
    return df

def detrend_bag(df, single=False):
    """
    Remove the linear effect in the brain age gap (BAG) for each participant in the dataframe.
    Input:
        df: dataframe including true and predicted ages & calculated BAGs
        single: Boolean indicating whether linear effect should be removed for
                a single (=TRUE) ICA modality or both (=FALSE).
    Output:
        df: dataframe with additional detrended BAG column(s)
    """
    X = np.array(df.loc[:,'age'], ndmin=2)
    X = np.reshape(X, (-1,1))
    if single:
        y = df.loc[:,'bag']
        model = LinearRegression()
        model.fit(X, y)
        trend = model.predict(X)
        df.loc[:,'bag_detrended'] = y-trend
    else: 
        # ICA25
        y_25 = df.loc[:,'bag_ICA25']
        model_25 = LinearRegression()
        model_25.fit(X, y_25)
        trend_25 = model_25.predict(X)
        df.loc[:,'bag_ICA25_detrended'] = y_25-trend_25
        # ICA100
        y_100 = df.loc[:,'bag_ICA100']
        model_100 = LinearRegression()
        model_100.fit(X, y_100)
        trend_100 = model_100.predict(X)
        df.loc[:,'bag_ICA100_detrended'] = y_100-trend_100
    return df
    
def strip_network_names(name, remove_hemisphere=False):
    """
    Strips long Schaefer parcellation network names to contain only the pure network names.
    Input:
        name: network name (str), e.g. '7Networks_LH_Vis_1'.
        remove_hemisphere: Boolean flag. If True, strips LH_/RH_, too. Default: False.
    Output:
        name: stripped network name (str), e.g. 'LH_Vis' (or 'Vis' if remove_hemisphere = True).
    """
    pattern_l = r'^\d+Networks_'
    pattern_r = r'_\d+$'
    if remove_hemisphere:
        name = re.sub(pattern_l+'[L|R]H_','',name)
    else:
        name = re.sub(pattern_l,'',name)
    name = re.sub(pattern_r,'',name)
    return name

def collect_predictions(predictions):
    """
    Collects all batched predictions and their corresponding sub IDs 
    from CNNs into one concise list each.
    Input: 
        predictions: list of tensors 
            ([[batch 1 [ids], [preds]], [batch 2 [ids], [preds]],...])
    Output:
        all_ids: 1D list of all subject IDs
        all_preds: 1D list of all age predictions
    """
    all_ids = []
    all_preds = []
    for batch in range(len(predictions)):
        all_ids += predictions[batch][0].tolist()
        all_preds += predictions[batch][1].squeeze().tolist()
    return all_ids, all_preds

def get_true_ys(label_df, ids_list):
    return [int(label_df.loc[label_df['eid']==id]['age']) for id in ids_list]

def load_lr_scheduler_config(path, optimizer):
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    lr_scheduler_config = dict()
    if config['scheduler'] == 'ReduceLROnPlateau':
        lr_scheduler_config['scheduler'] = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, 
                                                                                      mode=config['mode'], 
                                                                                      factor=config['factor'], 
                                                                                      patience=config['patience'], 
                                                                                      threshold=config['threshold'], 
                                                                                      cooldown=config['cooldown'])
        lr_scheduler_config['interval'] = config['interval']
        lr_scheduler_config['frequency'] = config['frequency']
        lr_scheduler_config['monitor'] = config['monitor']
    elif config['scheduler'] == 'OneCycleLR':
        lr_scheduler_config['scheduler'] = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer,
                                                                               max_lr=config['max_lr'],
                                                                               total_steps=config['total_steps'])
    lr_scheduler_config['name'] = config['scheduler']
    return lr_scheduler_config

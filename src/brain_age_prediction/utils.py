import os
import re
import random
import logging
from typing import Optional
import yaml
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.utils import resample

# Pytorch etc
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.utilities.rank_zero import rank_zero_warn

# visualisation
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

# import data + model modules
from brain_age_prediction import data, models, viz

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

def save_heldout_data_info(path_filename, datamodule):
    """
    Saves overview of datapoints (participant ID, age) to specified save path.
    """
    datamodule.data.labels.to_csv(path_filename, index=False)

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

# copy _select_seed_randomly and seed_everything function provided in Lightning vers. 1.6.3
# that is not available anymore in 2.0.2 from 
# https://pytorch-lightning.readthedocs.io/en/1.6.3/_modules/pytorch_lightning/utilities/seed.html#seed_everything
log = logging.getLogger(__name__)

def _select_seed_randomly(min_seed_value: int = 0, max_seed_value: int = 255) -> int:
    return random.randint(min_seed_value, max_seed_value)

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
        data_path: path to location where data is saved (expectations see data.DataModule)
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
    variable_CNN = models.variable1DCNN(in_channels=config['parameters']['in_channels'],
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
    datamodule = data.UKBBDataModule(data_path,
                                         ica=config['parameters']['ica'],
                                         good_components=config['parameters']['good_components'])

    # train model
    trainer.fit(variable_CNN, datamodule=datamodule)
    print('Training complete.')

    # save info on which data was used + what the train/val/test split was
    save_data_info(path=full_log_path, datamodule=datamodule)
    
    return trainer, variable_CNN, datamodule

def predict_w_model(trainer, model, datamodule, log_path, output=True):
    """
    Function for predicting the brain age of subjects in the validation + test set
    using a previously trained model. Automatically saves predictions in data_info.
    Input:
        trainer: the trainer instance of the model model
        model: the trained model
        datamodule: PyTorch Lightning UKBB DataModule instance
        log_path: path where to save the predictions
        output: Boolean flag whether to return the predictions dataframe
    Output:
        preds_df: dataframe with predicted ages
    """
    # model.eval()
    predictions = trainer.predict(model, datamodule)
    preds_df = pd.DataFrame(columns=['eid','batch_nb','predicted_age'])
    count = 0
    for batch_number, batch in enumerate(predictions):
        for i in range(len(batch[0])):
            preds_df.loc[count,'eid'] = int(batch[0][i])
            preds_df.loc[count,'batch_nb'] = batch_number
            preds_df.loc[count,'predicted_age'] = float(batch[1][i])       
            count += 1
    
    preds_df.to_csv(log_path, index=False)
    if output:
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
    viz.plot_training(data=metrics, title=f'Training visualisation of the ICA{ica_info} 1D-CNN with {gc} {data_info}.')

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

def get_data_overview_with_splitinfo(path_to_data_info, heldout_path=None):
    """
    Loads information from the data_info directory about participants' ages 
    and in which split participants were present during model training/testing.
    Input:
        path_to_data_info: path to where data_info is saved (without data_info itself in path).
        heldout_path: if applicable, path to heldout test set overview (including file name).
    Output:
        data_overview: participant/age/split overview as dataframe  
    """
    data_overview = load_datainfo(path_to_data_info, 'overview')
    train_idx = load_datainfo(path_to_data_info, 'train_idx')
    val_idx = load_datainfo(path_to_data_info, 'val_idx')
    test_idx = load_datainfo(path_to_data_info, 'test_idx')
    
    # add split information to items in data_overview based on sub ID
    for idx in train_idx:
        data_overview.loc[data_overview['eid'] == idx, 'split'] = 'train'
    for idx in val_idx:
        data_overview.loc[data_overview['eid'] == idx, 'split'] = 'val'
    for idx in test_idx:
        data_overview.loc[data_overview['eid'] == idx, 'split'] = 'test'
    if heldout_path:
        heldout_overview = pd.read_csv(heldout_path)
        heldout_overview['split'] = 'heldout_test'
        data_overview = pd.concat([data_overview, heldout_overview], ignore_index=True)
            
    return data_overview

def get_metadata(ukbb_data_path):
    """
    Get additional information about participants used in model training/testing; 
    also add in which split a participant was present.
    Input:
        ukbb_data_path: path to ukb_data directory (ukb_data included). E.g. 'ritter/share/data/UKBB/ukb_data'.
    Output:
        meta_df: overview dataframe containing the IDs of included participants + additional metadata.
    """    
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
    return meta_df

def merge_metadata_with_splitinfos(ukbb_data_path, path_to_data_info, heldout_path=None):
    """
    Generate big overview, retrieving + merging info of participant ID's, their ages, which split they
    occurred in, and health-related meta data.
    Input:
        ukbb_data_path: path to ukb_data directory (ukb_data included). E.g. 'ritter/share/data/UKBB/ukb_data'.
        path_to_data_info: path to where data_info is saved (without data_info itself in path).
        heldout_path: if applicable, path to heldout test set indices (including file name).
    Output:
        meta_df: overview dataframe containing the IDs of included participants + which split they occured in
            + additional metadata.

    """
    meta_df = get_metadata(ukbb_data_path)
    data_overview = get_data_overview_with_splitinfo(path_to_data_info, heldout_path)
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

def load_test_batches(train_ratio=0.88, batch_size_dev=2000, batch_size_eval=5000):
    """
    Loads + returns a batch each of the development and heldout test sets.
    Loads entire test sets with default batch sizes.
    For more info see load_test_batch.
    """
    dev_batch = load_test_batch(dev=True, train_ratio=train_ratio, batch_size=batch_size_dev)
    eval_batch = load_test_batch(dev=False, batch_size=batch_size_eval)
    return dev_batch, eval_batch

def load_test_batch(dev=True, train_ratio=0.88, batch_size=2000):
    """
    Loads + returns a batch of a test set.
    Input:
        dev: Boolean flag whether to use development or heldout set.
            If True, dev test set is loaded; if False, heldout test
            set is loaded. Default: True.
        train_ratio: parameter for train/val/test split regulation. 
            On a scale from 0 to 1, which proportion of data is to 
            be used for training? Default: 0.88.
        batch_size: how many subs of test set to load.
            With 2000, dev set is loaded entirely; with 5000, heldout
            set is loaded entirely. Smaller batch sizes load a subset.
            Default: 2000.
    Output:
        batch: [[subjects' timeseries], [subjects' ages], [subjects' ID's]]
    """
    datamodule = data.UKBBDataModule(dev=dev, train_ratio=train_ratio, batch_size=batch_size)
    datamodule.setup()
    return next(iter(datamodule.test_dataloader()))

def get_network_names(network_names_path='../../data/schaefer/7n100p/label_names_7n100p.csv',
                      remove_nr=False, remove_hemisphere=False):
    """
    Get list of all brain area / network names.
    Options to remove the network number and/or the hemisphere affiliation.
    Input:
        network_names_path: path to relevant label_names_[VARIANT].csv.
        remove_nr: Boolean flag. If True, strips numbering of networks (_1, _2 etc.). Default: True.
        remove_hemisphere: Boolean flag. If True, strips LH_/RH_, too. Default: False.
    Returns:
        brain_areas: list of network names as strings.
    """
    network_names_path = network_names_path
    network_names_df = pd.read_csv(network_names_path, names=['network_name'])
    brain_areas = network_names_df['network_name'].values
    brain_areas = [strip_network_names(brain_areas[i], remove_nr=remove_nr, remove_hemisphere=remove_hemisphere)
                   for i in range(len(brain_areas))]
    return brain_areas

#### OTHER HELPER FUNCTIONS
def calculate_bag(df, models=None):
    """
    Calculate the brain age gap (BAG) for each participant in the dataframe.
    Input:
        df: dataframe including true and predicted ages
        models: list of model names for which predicted ages exist as
                "predicted_age_modelname" column. BAGs will be calculated for
                each model's predictions. If None, a single "predicted_age"
                column is expected for which to calculate BAGs.
    Output:
        df: dataframe with additional BAG column(s)
    """
    if not models:
        df['bag'] = df['predicted_age'] - df['age']
    else:
        for model in models:
            df['bag_'+model] = df['predicted_age_'+model] - df['age']
    return df

def detrend_bag(df, models=None):
    """
    Remove the linear effect in the brain age gap (BAG) for each participant in the dataframe.
    Input:
        df: dataframe including true and predicted ages & calculated BAGs
        models: list of model names for which calculated BAGs exist as
                "bag_modelname" column. Linear effects will be removed for
                each model's BAGs. If None, a single "bag"
                column is expected for which to detrend BAGs.
    Output:
        df: dataframe with additional detrended BAG column(s)
    """
    X = np.array(df.loc[:,'age'], ndmin=2)
    X = np.reshape(X, (-1,1))
    if not models:
        y = df.loc[:,'bag']
        model = LinearRegression()
        model.fit(X, y)
        trend = model.predict(X)
        df.loc[:,'bag_detrended'] = y-trend
    else: 
        for model_name in models:
            y = df.loc[:,'bag_'+model_name]
            model = LinearRegression()
            model.fit(X, y)
            trend = model.predict(X)
            df.loc[:,'bag_'+model_name+'_detrended'] = y-trend
    return df

def preds_corr(df, column=None, model_name=None, variables=True):
    """
    Calculate correlations between all aspects of interest.
    Expects an overview dataframe that is limited to those IDs for which
    predictions exist.
    Input:
        df: (heldout) overview pandas dataframe.
        column: necessary if variables=True. Name for column to correlate 
            with meta information variables, e.g. 'bag' or 'bag_detrended'.
        model_name: necessary if variables=False. Model name for which to 
            calculate correlations between the true age and the model's 
            predicted age, BAG, and detrended BAG.
        variables: Boolean flag. If variables=True, calculate correlations 
            between the models' BAG + degrended BAG and all health 
            variables of interest. If variables=False, calculate correlations
            between the true age and the models' predicted age + BAG + 
            detrended BAG.
        models: list of model names for which BAGs exist as "bag_modelname"
                and "bag_modelname_detrended" columns. If None, single 
                "bag"/"bag_detrended" columns are expected for which to 
                calculate correlations.
    Output:
        corrs: list of correlations.
    """
    var_columns = ['bmi', 'digit substitution', 'education', 'fluid intelligence',
                   'grip', 'depressive episode', 'all depression',
                   'recurrent depressive disorder', 'multiple sclerosis', 'sex',
                   'weekly beer', 'genetic pc 1', 'genetic pc 2', 'genetic pc 3']
    corrs = []
    if variables:
        for var_column in var_columns:
            corrs.append(df[column].corr(df[var_column], method='spearman'))
    else:
        columns = []
        if model_name:
            columns.append(f'predicted_age_{model_name}')
            columns.append(f'bag_{model_name}')
            columns.append(f'bag_{model_name}_detrended')
            model_name = '_'+model_name
        else:
            columns = ['predicted_age', 'bag', 'bag_detrended']
        for var_column in columns:
            corrs.append(df['age'].corr(df[var_column], method='spearman'))
    return corrs

def bootstrap_corrs(df, model_name=None, n_iterations=10):
    """
    Bootstrap prediction overviews for one model.
    Input:
        df: (heldout) overview pandas dataframe.
        model_name: model name for which to calculate correlations.
        n_iterations: number of bootstrapping iterations. Default: 10.
    Output:
        corrs_dict: nested dictionary containing mean + std of all
            correlations of interest.
    """
    if model_name:
        bag = 'bag_'
    else:
        bag = 'bag'
        model_name = ''
    # collect correlations
    corrs_true_age = []
    corrs_vars = []
    corrs_vars_detrended = []
    for i in range(n_iterations):
        bootstrapped_df = resample(df, replace=True, random_state=i)
        corrs_true_age.append(preds_corr(bootstrapped_df,model_name=model_name,variables=False))
        corrs_vars.append(preds_corr(bootstrapped_df,column=bag+model_name,variables=True))
        corrs_vars_detrended.append(preds_corr(bootstrapped_df,column=bag+model_name+'_detrended',variables=True))
    corrs_dict = {
        'corrs true age': {'mean': np.mean(corrs_true_age, axis=0),
                           'std': np.std(corrs_true_age, axis=0)},
        'corrs variables': {'mean': np.mean(corrs_vars, axis=0),
                            'std': np.std(corrs_vars, axis=0)},
        'corrs variables detrended': { 'mean': np.mean(corrs_vars_detrended, axis=0),
                                      'std': np.std(corrs_vars_detrended, axis=0)}
    }
    return corrs_dict

def bootstrap_pipeline(df, models=None, n_iterations=10):
    """
    """
    if not models:
        models = ['']
    
    # get overviews for first model
    corrs_true_age_df, corrs_vars_df = get_model_bootstrap_overview(df=df,
                                                                    model_name=models[0],
                                                                    n_iterations=n_iterations)
    # optional further models
    if len(models) > 1:
        for model_idx in range(1, len(models)):
            new_corrs_true_age_df, new_corrs_vars_df = get_model_bootstrap_overview(df=df,
                                                                                    model_name=models[model_idx],
                                                                                    n_iterations=n_iterations)
            # merge with previous overview
            corrs_true_age_df = pd.concat([corrs_true_age_df,new_corrs_true_age_df], ignore_index=True)
            corrs_vars_df = corrs_vars_df.merge(new_corrs_vars_df, on='Variable')
    return corrs_true_age_df, corrs_vars_df
    
def get_model_bootstrap_overview(df, model_name, n_iterations):
    """
    Create overviews over true age + variable correlations for one model,
    displaying the data's 'raw' correlations as well as 
    the bootstrap mean + SEM, and the zscore.
    """
    # heldout corrs
    corrs_true_age_df = viz.preds_corr_overview(df, variables=False, models=[model_name])
    corrs_vars_df = viz.preds_corr_overview(df, variables=True, models=[model_name])
    # bootstrapped corrs
    corrs_bs_dict = bootstrap_corrs(df=df, model_name=model_name, n_iterations=n_iterations)
    corrs_true_age_bs_df = viz.bootstrap_overview(corrs_bs_dict, variables=False, model=model_name)
    corrs_vars_bs_df = viz.bootstrap_overview(corrs_bs_dict, variables=True, model=model_name)
    # merge
    corrs_true_age_df = corrs_true_age_df.merge(corrs_true_age_bs_df, on='True age vs.')
    corrs_vars_df = corrs_vars_df.merge(corrs_vars_bs_df, on='Variable')
    # calculate zscore
    corrs_true_age_df['Corr z'] = corrs_true_age_df['Corr mean'] / corrs_true_age_df['Corr sem']
    corrs_vars_df['Corr BAG '+model_name+' z'] = corrs_vars_df['Corr BAG '+model_name+' model mean'] / corrs_vars_df['Corr BAG '+model_name+' model sem']
    corrs_vars_df['Corr detrended BAG '+model_name+' z'] = corrs_vars_df['Corr detrended BAG '+model_name+' model mean'] / corrs_vars_df['Corr detrended BAG '+model_name+' model sem']
    # reorder colums
    cols = ['Variable','Corr BAG '+model_name+' model', 'Corr BAG '+model_name+' model mean',
            'Corr BAG '+model_name+' model sem', 'Corr BAG '+model_name+' z',
            'Corr detrended BAG '+model_name+' model', 'Corr detrended BAG '+model_name+' model mean', 
            'Corr detrended BAG '+model_name+' model sem', 'Corr detrended BAG '+model_name+' z']
    corrs_vars_df = corrs_vars_df.reindex(columns=cols)
    return corrs_true_age_df, corrs_vars_df

def strip_network_names(name, remove_nr=True, remove_hemisphere=False):
    """
    Strips long Schaefer parcellation network names to contain only the pure network + area names.
    Input:
        name: network name (str), e.g. '7Networks_LH_Vis_1'.
        remove_nr: Boolean flag. If True, strips numbering of networks (_1, _2 etc.). Default: True.
        remove_hemisphere: Boolean flag. If True, strips LH_/RH_, too. Default: False.
    Output:
        name: stripped network name (str), e.g. 'LH_Vis' (or 'Vis' if remove_hemisphere = True).
    """
    pattern_l = r'^\d+Networks_'
    if remove_hemisphere:
        name = re.sub(pattern_l+'[L|R]H_','',name)
    else:
        name = re.sub(pattern_l,'',name)
    if remove_nr:
        pattern_r = r'_\d+$'
        name = re.sub(pattern_r,'',name)
    return name

def extract_hemisphere(full_networ_str):
    """
    Extracts hemisphere from network name of the form 'HEMISPHERE_NETWORK_(AREA_)NR'.
    Input:
        full_network_str: full network name (str), e.g. 'LH_Vis_1'.
    Output:
        hemisphere: hemisphere info (str), e.g. 'LH'.
    """
    pattern_r = r'_.*'
    hemisphere = re.sub(pattern_r,'',full_networ_str)
    return hemisphere

def extract_network(full_network_str):
    """
    Extracts pure network name from network name of the form 'HEMISPHERE_NETWORK_(AREA_)NR'.
    Input:
        full_network_str: full network name (str), e.g. 'LH_Vis_1'.
    Output:
        network_name: pure network name (str), e.g. 'Vis'.
    """
    pattern_l = r'[L|R]H_'
    pattern_r = r'_.*\d+$'
    network_name = re.sub(pattern_l,'',full_network_str)
    network_name = re.sub(pattern_r,'',network_name)
    return network_name

def extract_area(full_network_str):
    """
    Extracts pure area name from network name of the form 'HEMISPHERE_NETWORK_(AREA_)NR'.
    Input:
        full_network_str: full network name (str), e.g. 'LH_DorsAttn_Post_1'.
    Output:
        area_name: pure area name (str), e.g. 'Post'.
    """
    network_name = extract_network(full_network_str)
    if network_name == 'Vis':
        area_name = 'full Vis'
    elif network_name == 'SomMot':
        area_name = 'full SomMot'
    else:
        pattern_l = r'^[L|R]H_'+network_name+'_'
        pattern_r = r'_\d+$'
        area_name = re.sub(pattern_l,'',full_network_str)
        area_name = re.sub(pattern_r,'',area_name)
        if area_name in ['PFCl','Par']:
            area_name = area_name+' '+network_name
    return area_name

def add_specific_network_columns(df):
    """
    Insert information columns on a parcellation's hemisphere, network and brain area
    to an existing long-form dataframe.
    Input:
        df: long-form dataframe with at least the following columns: 
            'id', 'parcellation' (, ...)
    Output:
        df: long-form dataframe with at least the following columns: 
            'id', 'parcellation', 'hemisphere', 'network', 'area' (, ...)
    """
    df.insert(2,'hemisphere',df.apply(lambda row: extract_hemisphere(row['parcellation']), axis=1))
    df.insert(3,'network',df.apply(lambda row: extract_network(row['parcellation']), axis=1))
    df.insert(4,'area',df.apply(lambda row: extract_area(row['parcellation']), axis=1))
    return df

def create_long_df(shap_values, sub_shortcut_path='../../data/schaefer/heldout_test_set_100-500p_dataloader_order_43.csv'):
    """
    Create long table of mean SHAP values per parcellation for easier visualisation creation.
    Input:
        shap_values: array of SHAP values of shape (N_SUBS, 100, 490).
        sub_shortcut_path: path to shortcut of subject ordering from the dataloader 
            that was used to create the SHAP values.
    Output:
        shap_df: long-form table of mean SHAP values per parcellation per subject.
    """
    # load order of subjects
    sub_order = np.loadtxt(sub_shortcut_path, dtype=int)
    # get mean SHAP value per parcellation for each sub
    mean_area = np.mean(np.abs(shap_values),axis=2)
    # get brain area / network names
    network_names = get_network_names()
    # create wide df
    shap_df = pd.DataFrame(mean_area, columns=network_names)
    shap_df.insert(0,'id',sub_order)
    # convert to long df
    shap_df = shap_df.melt(id_vars=['id'], var_name='parcellation', value_name='shap')
    # add column of absolute SHAP values
    shap_df = add_specific_network_columns(shap_df)
    return shap_df

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
    make_reproducible()
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
    elif config['scheduler'] == 'OneCycleLR':
        lr_scheduler_config['scheduler'] = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer,
                                                                               max_lr=config['max_lr'],
                                                                               total_steps=config['total_steps'])
    lr_scheduler_config['interval'] = config['interval']
    lr_scheduler_config['frequency'] = config['frequency']
    lr_scheduler_config['monitor'] = config['monitor']
    lr_scheduler_config['name'] = config['scheduler']
    return lr_scheduler_config

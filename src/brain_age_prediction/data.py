# standard modules
import numpy as np
import pandas as pd
from scipy.stats import zscore
import math
import random

# PyTorch modules
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import pytorch_lightning as pl

# own utils module
from brain_age_prediction import utils

##################################################################################

class UKBB_ICA_ts(Dataset):
    """
    Dataset class that prepares ICA timeseries of rs-fMRI data from the UKBioBank.
    Input:
        data_path: path to ukb_data directory (ukb_data included). E.g. 'ritter/share/data/UKBB/ukb_data'.
        ica: whether to load ICA25 ('25') or ICA100 ('100'). Expects string. Default: 25.
        good_components: boolean flag to indicate whether to use only the good components or all components. Default: False.
        all_data: boolean flag to indicate whether to use all data or only a subset of 100 samples. Default: True.
    """
    def __init__(self, data_path, ica='25', good_components=False, all_data=True): 
        # save data path + settings
        self.data_path = data_path
        self.ica = ica
        self.good_components = good_components
        self.all_data = all_data
        self.additional_data = '../../data/ica/'

        # catch ica error
        if ica != '25' and ica != '100':
            # ERROR MESSAGE ODER SO
            pass

        # get indices of good components
        if good_components:
            if ica == '25':
                self.good_components_idx = np.loadtxt(self.additional_data+'rfMRI_GoodComponents_d25_v1.txt', dtype=int)
            else:
                self.good_components_idx = np.loadtxt(self.additional_data+'rfMRI_GoodComponents_d100_v1.txt', dtype=int)
            # component numbering starts at 1, not at 0
            self.good_components_idx = self.good_components_idx-1
            
        # META INFORMATION
        # get target information (age)
        age_df = pd.read_csv(self.data_path+'table/targets/age.tsv', sep='\t', names=['age'])
        # get subject IDs, rename column for later merge
        ids_df = pd.read_csv(self.data_path+'table/ukb_imaging_filtered_eids.txt')
        ids_df.rename(columns={'f.eid': 'eid'}, inplace=True)
        # get location of ica files
        location_df = pd.read_csv(self.additional_data+'ica_locations.csv')
        # combine information
        # first ids and age
        meta_df = pd.concat([ids_df, age_df], axis=1)
        # merge with location info based on eid
        self.location_column = 'rfmri_ica'+self.ica+'_ts'
        meta_df = meta_df.merge(location_df[['eid',self.location_column]], on='eid', how='left')
        # limit df to available ICA25 data points
        meta_df = meta_df[meta_df[self.location_column].isna()==False]
        # reset index
        meta_df = meta_df.reset_index(drop=True, inplace=False)
        
        # reduce amount of data for debugging etc.
        if not self.all_data:
            meta_df = meta_df[0:100]
        
        # save meta_df information as labels
        self.labels = meta_df
        
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, sub_id):       
        # get label (age)
        label = self.labels.loc[self.labels['eid'] == sub_id, 'age'].values[0]
        # get filname/path to timeseries
        ts_path = self.labels.loc[self.labels['eid'] == sub_id,self.location_column].values[0]
        
        # load + standardise timeseries
        ## if good_components = True, only load good components
        if self.good_components:
            timeseries = np.loadtxt(ts_path, 
                                    usecols=tuple(self.good_components_idx),
                                    max_rows=490) # crop longer input to 490 rows
        else:
            timeseries = np.loadtxt(ts_path, 
                                    max_rows=490) # crop longer input to 490 rows
        ## change axes ([components, time points])
        timeseries = np.swapaxes(timeseries, 1, 0)
        ## 
        ## standardise each component's timeseries
        timeseries = zscore(timeseries, axis=1)
        
        # turn data into tensors
        timeseries = torch.from_numpy(timeseries)
        timeseries = timeseries.float()
        label = torch.tensor(label)
        label = label.float()
        return timeseries, label, sub_id        

class UKBB_Schaefer_ts(Dataset):
    """
    Dataset class that prepares Schaefer timeseries of rs-fMRI data from the UKBioBank.
    Input:
        data_path: path to ukb_data directory (ukb_data included). E.g. 'ritter/share/data/UKBB/ukb_data'.
        schaefer_variant: which variant to load. Expects string of the form 'XnYp' where X is the number of
            networks and p is the number of parcellations. Possible values: '7n100p','7n200p','7n500p',
            '7n700p','7n1000p','17n100p','17n200p','17n500p','17n700p','17n1000p'. Default: '7n100p'.
        shared_variants: list of Schaefer variants which share usable subject IDs. 
            Default: ['7n100p','7n200p','7n500p','17n100p','17n200p','17n500p']
        corr_matrix: boolean flag to determine whether to use the ordinary Schaefer timeseries (=False)
            or a subject-wise functional connectivity correlation matrix (=True). Default: False.
        corr_kind: Type of correlation coefficients to calculate if corr_matrix=True. Default: 'pearson'.
        additional_data_path: path to additional data info directory. Default: '../../data/schaefer/'.
        heldout_set_name: name of held-out test set IDs created with `define_heldout_test_set.py` that are 
            totally removed from the initial training/testing of models. 
            Default: 'heldout_test_set_100-500p.csv'.
        all_data: boolean flag to indicate whether to use all data or only a subset of 100 samples. Default: True.
        dev: boolean flag to indicate whether model training/testing is still in the development phase. If True,
            held-out IDs are dropped from meta_df, if False, only held-out IDs are kept. Default: True.
    """
    def __init__(self, data_path, schaefer_variant='7n100p', shared_variants=['7n100p','7n200p','7n500p','17n100p','17n200p','17n500p'], corr_matrix=False, corr_kind='pearson', additional_data_path='../../data/schaefer/', heldout_set_name='heldout_test_set_100-500p.csv', all_data=True, dev=True):
        # save data path + settings
        self.data_path = data_path
        self.schaefer_variant = schaefer_variant
        self.shared_variants = shared_variants
        self.corr_matrix = corr_matrix
        self.corr_kind = corr_kind
        self.additional_data_path = additional_data_path
        self.heldout_path = additional_data_path+heldout_set_name
        self.all_data = all_data
        self.dev = dev

        possible_variants = ['7n100p','7n200p','7n500p','7n700p','7n1000p',
                             '17n100p','17n200p','17n500p','17n700p','17n1000p']
        
        assert self.schaefer_variant in possible_variants, 'Given variant does not exist. Check typos!'
        
        # get ID infos for relevant variants
        usable_ids = set(utils.get_usable_schaefer_ids(schaefer_data_dir=self.additional_data_path, variants=shared_variants))
        heldout_ids = set(np.loadtxt(self.heldout_path, dtype=int))

        # META INFORMATION
        # get target information (age)
        age_df = pd.read_csv(self.data_path+'table/targets/age.tsv', sep='\t', names=['age'])
        # get subject IDs, rename column 
        ids_df = pd.read_csv(self.data_path+'table/ukb_imaging_filtered_eids.txt')
        ids_df.rename(columns={'f.eid': 'eid'}, inplace=True)
        # combine information
        meta_df = pd.concat([ids_df, age_df], axis=1)
        # only keep relevant IDs
        if self.dev:
            # limit df to available data points
            meta_df = meta_df[meta_df['eid'].isin(usable_ids)]
            # drop held-out IDs
            meta_df.drop(meta_df[meta_df['eid'].isin(heldout_ids)].index, inplace=True)
        else:
            # keep only held-out IDs
            meta_df = meta_df[meta_df['eid'].isin(heldout_ids)]
        # reset index
        meta_df.reset_index(drop=True, inplace=True)
        
        # reduce amount of data for debugging etc.
        if not self.all_data:
            meta_df = meta_df[0:5000]
        
        # save meta_df information as labels
        self.labels = meta_df
        
        
    def __len__(self):
        return len(self.labels)
    
    #def to_tensor(data, label):
     #   """
      #  Make input + label PyTorch compatible by turning arrays into tensors.
       # """
        #data = torch.from_numpy(data)
        #label = torch.tensor(label)
        #return data.float(), label.float()
    
    def __getitem__(self, sub_id):       
        # get label (age)
        label = self.labels.loc[self.labels['eid'] == sub_id, 'age'].values[0]
        # get filname/path to timeseries
        ts_path = self.data_path+'bids/sub-'+str(sub_id)+'/ses-2/func/sub-'+str(sub_id)+'_ses-2_task-rest_Schaefer'+self.schaefer_variant+'.csv.gz'
        
        # load + standardise timeseries
        # don't include the first column that names the networks/parcellations 
        timeseries = np.loadtxt(ts_path, skiprows=1, usecols=tuple([i for i in range(1,491)]), delimiter=',') 
        # standardise each component's timeseries
        timeseries = zscore(timeseries, axis=1)
        
        if self.corr_matrix:
            # calculate correlations
            if self.corr_kind == 'pearson':
                correlation_matrix = np.corrcoef(timeseries)
            else:
                raise NameError(f"Correlation kind {self.corr_kind} is not defined")        
            # turn data into tensors
            #model_input, label = self.to_tensor(correlation_matrix, label)
            correlation_matrix = torch.from_numpy(correlation_matrix)
            model_input = correlation_matrix.float()
            label = torch.tensor(label)
            label = label.float()
        else:
            # turn data into tensors
            #model_input, label = self.to_tensor(timeseries, label)
            timeseries = torch.from_numpy(timeseries)
            model_input = timeseries.float()
            label = torch.tensor(label)
            label = label.float()
        
        return model_input, label, sub_id        

##################################################################################   
class UKBBDataModule(pl.LightningDataModule):
    """
    Pytorch Lightning style DataModule class that prepares & loads timeseries
    of rs-fMRI data from the UKBioBank.
    Input:
        dataset_type: which torch Dataset to use for loading data. This determins the data modality 
            (ICA ts ("UKBB_ICA_ts") or Schaefer ts ("UKBB_Schaefer_ts")) and consequently which input variables have an effect.
        data_path: path to ukb_data directory (ukb_data included). E.g. 'ritter/share/data/UKBB/ukb_data'.
        ica: whether to load ICA25 ('25') or ICA100 ('100'). Expects string. Default: 25.
        good_components: boolean flag to indicate whether to use only the good components or all components. Default: False.
        schaefer_variant: which variant to load. Expects string of the form 'XnYp' where X is the number of
            networks and p is the number of parcellations. Possible values: '7n100p','7n200p','7n500p',
            '7n700p','7n1000p','17n100p','17n200p','17n500p','17n700p','17n1000p'. Default: '7n100p'.
        shared_variants: list of Schaefer variants which share usable subject IDs. 
            Default: ['7n100p','7n200p','7n500p','17n100p','17n200p','17n500p']
        corr_matrix: boolean flag to determine whether to use the ordinary Schaefer timeseries (=False)
            or a subject-wise functional connectivity correlation matrix (=True). Default: False.
        corr_kind: Type of correlation coefficients to calculate if corr_matrix=True. Default: 'pearson'.
        additional_data_path: path to additional data info directory. Default: '../../data/schaefer/'.
        heldout_set_name: name of held-out test set IDs created with `define_heldout_test_set.py` that are 
            totally removed from the initial training/testing of models. 
            Default: 'heldout_test_set_100-500p.csv'.
        all_data: boolean flag to indicate whether to use all data or only a subset of 100 samples. Default: True.
        dev: boolean flag to indicate whether model training/testing is still in the development phase. If True,
            held-out IDs are dropped from meta_df, if False, only held-out IDs are kept. Default: True.
        batch_size: batch size for DataLoaders. Default: 128.
        seed: random seed that is used. Default: 43.
        train_ratio: first parameter for train/val/test split regulation. 
            On a scale from 0 to 1, which proportion of data is to be used for training? Default: 0.88.
        val_test_ratio: second parameter for train/val/test split regulation.
            On a sclae form 0 to 1, which proportion of the split not used for training is to be used for 
            validating/testing? >0.5: more data for validation; <0.5: more data for testing. Default: 0.5.
    """
    def __init__(self, dataset_type='UKBB_Schaefer_ts', data_path='/ritter/share/data/UKBB/ukb_data/',
                 ica='25', good_components=False, 
                 schaefer_variant='7n100p', shared_variants=['7n100p','7n200p','7n500p','17n100p','17n200p','17n500p'],
                 corr_matrix=False, corr_kind='pearson',
                 additional_data_path='../../data/schaefer/', heldout_set_name='heldout_test_set_100-500p.csv', 
                 all_data=True, dev=True, 
                 batch_size=128, seed=43, train_ratio=0.8, val_test_ratio=0.5): 
        super().__init__()
        self.save_hyperparameters()
        self.data_path = data_path
        self.dataset_type = dataset_type
        self.ica = ica
        self.good_components = good_components
        self.schaefer_variant = schaefer_variant
        self.shared_variants = shared_variants
        self.corr_matrix = corr_matrix
        self.corr_kind = corr_kind
        self.additional_data_path = additional_data_path
        self.heldout_set_name = heldout_set_name
        self.all_data = all_data
        self.dev = dev
        self.batch_size = batch_size
        self.seed = seed
        self.train_ratio = train_ratio
        self.val_test_ratio = val_test_ratio
        self.data = None
        # increase reproducibility
        utils.make_reproducible(seed)
        self.g = torch.Generator() # device='cuda'
        self.g.manual_seed(seed)
        
    def get_split_index(self, split_ratio, dataset_size):
        return int(np.floor(split_ratio * dataset_size))
   
    def setup(self, stage=None):
        utils.make_reproducible(self.seed)
        # runs on all GPUs
        # load data
        if self.dataset_type == 'UKBB_ICA_ts':
            self.data = UKBB_ICA_ts(self.data_path, self.ica, self.good_components, self.all_data) 
        elif self.dataset_type == 'UKBB_Schaefer_ts':
            self.data = UKBB_Schaefer_ts(self.data_path, self.schaefer_variant, self.shared_variants, self.corr_matrix, self.corr_kind, self.additional_data_path, self.heldout_set_name, self.all_data, self.dev) 

        dataset_size = self.data.labels['eid'].shape[0]
        indices = list(self.data.labels['eid'])  
            
        # development phase
        if self.dev:
            # split data   
            # shuffle
            np.random.shuffle(indices)
            indices = list(indices)
            # get list of split indices, feed to samplers
            split_index = self.get_split_index(self.train_ratio, dataset_size)
            self.train_idx, remain_idx = indices[:split_index], indices[split_index:]
            remain_split_index = self.get_split_index(self.val_test_ratio, len(remain_idx))
            self.val_idx, self.test_idx = remain_idx[:remain_split_index], remain_idx[remain_split_index:]

            assert len(self.val_idx)+len(self.test_idx) == len(self.val_idx+self.test_idx), 'Idx addition error'

            self.train_sampler = SubsetRandomSampler(self.train_idx, generator=self.g)
            self.val_sampler = SubsetRandomSampler(self.val_idx, generator=self.g)
            self.test_sampler = SubsetRandomSampler(self.test_idx, generator=self.g)
            # self.predict_sampler = SubsetRandomSampler(self.val_idx+self.test_idx, generator=self.g)
            self.predict_sampler_full = SubsetRandomSampler(self.val_idx+self.test_idx, generator=self.g)
            self.predict_sampler_val = SubsetRandomSampler(self.val_idx, generator=self.g)
            self.predict_sampler_test = SubsetRandomSampler(self.test_idx, generator=self.g)
        # final testing phase
        else:
            self.test_sampler = SubsetRandomSampler(indices, generator=self.g)
            self.predict_sampler_full = SubsetRandomSampler(indices, generator=self.g)
            
    
    def general_dataloader(self, data_part, data_sampler):
        data_loader = DataLoader(
                        data_part,
                        batch_size = self.batch_size,
                        sampler = data_sampler,
                        pin_memory = True, ##
                        num_workers = 0, ##
                        # drop_last=False,
                        worker_init_fn = utils.seed_worker,
                        generator = self.g,
                        )
        return data_loader
        
    def train_dataloader(self):
        return self.general_dataloader(self.data, self.train_sampler)
    
    def val_dataloader(self):
        return self.general_dataloader(self.data, self.val_sampler)
    
    def test_dataloader(self):
        return self.general_dataloader(self.data, self.test_sampler)
    
    def predict_dataloader(self):
        # return self.general_dataloader(self.data, self.predict_sampler)
        return self.general_dataloader(self.data, self.predict_sampler_full)
    

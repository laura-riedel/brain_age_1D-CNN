"""
Script for performing channel-wise occlusion for each subject in the heldout
test set and calculate the performance difference when predicting subjects'
ages with each data permutation using either the deep or shallow 1D-CNN.
"""
import argparse
import numpy as np
import h5py
import torch
from brain_age_prediction import data, models, utils

######################################################################
parser = argparse.ArgumentParser(prog='Occlusion',
                                description='Occlusion pipeline for the heldout test set.')
parser.add_argument('model', type=str,
                    help='Whether to generate explanations using the "deep" or "shallow" 1DCNN.')
parser.add_argument('-seed', type=int, default=43,
                    help='Set random seed. Default: 43.')
parser.add_argument('-heldout_path', type=str, default='/home/laurar/brain_age_1D-CNN/data/schaefer/heldout_test_set_100-500p.csv',
                    help='Path to heldout test set subject IDs to be used.')

args = parser.parse_args()
######################################################################
# save dir
save_dir = '/ritter/share/projects/laura_riedel_thesis/'
# paths to best model checkpoints
shallow_model_path = 'lightweight-brain-age-prediction/umd5tlvz/checkpoints/epoch=57-step=13108.ckpt'
deep_model_path = 'lightweight-brain-age-prediction/nx218mm3/checkpoints/epoch=26-step=6102.ckpt'

utils.make_reproducible(random_state=args.seed)

# initialise model
print('Initialise model...')
if args.model == 'shallow':
    model = models.shallow1DCNN.load_from_checkpoint(shallow_model_path)
elif args.model == 'deep':
    model = models.deep1DCNN.load_from_checkpoint(deep_model_path)
model.eval()

# initialise datamodule
datamodule = data.UKBBDataModule(dev=False)
datamodule.setup()

# load sub IDs
sub_ids = np.loadtxt(args.heldout_path, dtype=int)

i = 1
with h5py.File(save_dir+'occlusion_results_'+args.model+'.hdf5', 'a') as hdf5:
    for sub_id in sub_ids:
        print(f'Subject {i}/{len(sub_ids)}')

        # create hdf5 group
        grp = hdf5.create_group(str(sub_id))

        # get true age
        true_age = datamodule.data[sub_id][1].item()
        # get original timeseries
        orig_ts = datamodule.data[sub_id][0]
        # get original prediction
        with torch.no_grad():
            orig_pred = model(orig_ts.unsqueeze(dim=0)).item()
        # shuffled timeseries
        indices = torch.argsort(torch.rand_like(orig_ts), dim=-1)
        shuffled_ts = torch.gather(orig_ts, dim=-1, index=indices)

        # collect subject results
        # channel 0 at list[0], channel 1 at list[1] etc.
        occluded_preds = []
        pred_diffs = []
        occlusion_bags = []

        # channel-wise occlusion pipeline
        for channel in range(orig_ts.shape[0]):
            # occlude channel
            occluded_ts = orig_ts.detach().clone()
            occluded_ts[channel] = shuffled_ts[channel]
            # predict age with occlusion
            with torch.no_grad():
                occluded_pred = model(occluded_ts.unsqueeze(dim=0)).item()
            # calculate prediction difference
            # + update info
            occluded_preds.append(occluded_pred)
            pred_diffs.append(occluded_pred - orig_pred)
            occlusion_bags.append(occluded_pred - true_age)
        
        # add subject results as datasets to group
        grp.create_dataset('occluded preds', data=np.asarray(occluded_preds), compression='gzip', compression_opts=9)
        grp.create_dataset('pred diff', data=np.asarray(pred_diffs), compression='gzip', compression_opts=9)
        grp.create_dataset('occlusion BAG', data=np.asarray(occlusion_bags), compression='gzip', compression_opts=9)
        i += 1

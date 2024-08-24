import argparse
import h5py
import csv
import shap
from brain_age_prediction import models, utils
##################################################################################
parser = argparse.ArgumentParser(prog='Explainability',
                                description='Fits a DeepLiftSHAP explainer on development test data and generates and saves SHAP values for the heldout test set.')
parser.add_argument('model', type=str,
                    help='Whether to generate explanations using the "deep" or "shallow" 1DCNN.')
parser.add_argument('-save_name', type=str,
                    help='Under which name to save the shap values to HDF5.')
parser.add_argument('-batch_size_dev', type=int, default=2000,
                    help='Which size of the development test set to use. Default: 2000 (=all).')
parser.add_argument('-batch_size_eval', type=int, default=5000,
                    help='Which size of the heldout test set to use. Default: 5000 (=all).')
parser.add_argument('-save_sub_order_shortcut', type=bool, default=False,
                    help='Should a shortcut be saved in which order the subjects are loaded? (Only needed one time.) Default: False.')

args = parser.parse_args()
##################################################################################
# save dir
save_dir = '/ritter/share/projects/laura_riedel_thesis/'
# paths to best model checkpoints
shallow_model_path = 'lightweight-brain-age-prediction/umd5tlvz/checkpoints/epoch=57-step=13108.ckpt'
deep_model_path = 'lightweight-brain-age-prediction/nx218mm3/checkpoints/epoch=26-step=6102.ckpt'

utils.make_reproducible()

# load dev and heldout test sets
print('Load data...')
dev_test, eval_test = utils.load_test_batches(batch_size_dev=args.batch_size_dev, batch_size_eval=args.batch_size_eval)
# initialise model
print('Initialise model...')
if args.model == 'shallow':
    model = models.shallow1DCNN.load_from_checkpoint(shallow_model_path)
elif args.model == 'deep':
    model = models.deep1DCNN.load_from_checkpoint(deep_model_path)
model.eval()

# set up explainer with background of dev test set
print('Fit explainer instance...')
explainer = shap.DeepExplainer(model,dev_test[0])
# get shap values for heldout test set
print('Calculate SHAP values...')
shap_values = explainer.shap_values(eval_test[0])

# save predictions
print('Save SHAP values...')
with h5py.File(save_dir+'shap_values.hdf5', 'a') as hdf5:
    save_path = args.model+'/'+args.save_name
    hdf5.create_dataset(save_path, data=shap_values, compression='gzip', compression_opts=9)

# optional: save sub order shortcut
if args.save_sub_order_shortcut:
    print('Save shortcut in which order the subjects are loaded by the DataLoader...')
    sub_ids = eval_test[2].numpy()
    with open('../../data/schaefer/heldout_test_set_100-500p_dataloader_order_43.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for sub in sub_ids:
            writer.writerow([sub])
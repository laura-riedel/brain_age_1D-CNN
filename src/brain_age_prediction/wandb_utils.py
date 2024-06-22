# PyTorch modules
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

# custom module
from brain_age_prediction import data, models, utils, sklearn_utils

# other
import wandb
import sys
import shutil
from sklearn.metrics import r2_score

##################################################################################
# callbacks
def checkpoint_init():
    checkpoint = ModelCheckpoint(monitor='val_loss',
                                 save_top_k=1,
                                 mode='min')
    return checkpoint

# training a model
def wandb_train(config, name=None, tags=None, use_gpu=False, devices=None, dev=True, batch_size=128, max_epochs=None, 
                num_threads=1, seed=43, lr_scheduler_config_path=None, train_ratio=0.88, val_test_ratio=0.5, 
                save_datasplit=True, save_overview=False, all_data=True, test=False, finish=False, execution='nb'):
    """
    Function for training a model in a notebook using external config information. Logs to W&B.
    Optional trained model + datamodule output.
    Input:
        config: configuration dictionary containing meta information and parameters for 
            the model that shall be trained. Requested form:
            {project:..., group:..., name:..., tags:..., parameters:{...}}
        name: name for the W&B run. Defaults to the name in the config.
        tags: tags for the W&B run. Defaults to the tags in the config.
        use_gpu: Boolean flag indicating which accelerator to use for training. 
            Uses GPU if True and CPU if False. Default: False.
        devices: The devices to use when training; depends on accelerator. Default for GPU use: [1].
        dev: boolean flag to indicate whether model training/testing is still in the development phase. If True,
            held-out IDs are dropped from meta_df, if False, only held-out IDs are kept. Default: True.
        batch_size: batch size for DataLoaders. Default: 128.
        max_epochs: for how many epochs the model is supposed to train. Defaults to max_epochs in the config.
        num_threads: if CPU, how many CPU threads to use. Default: 1.
        seed: random seed that is used. Default: 43.
        lr_scheduler_config_path: path to the config file of a learning rate scheduler, if one wants to use one.
            Default: None.
        train_ratio: first parameter for train/val/test split regulation. 
            On a scale from 0 to 1, which proportion of data is to be used for training? Default: 0.88.
        val_test_ratio: second parameter for train/val/test split regulation.
            On a sclae form 0 to 1, which proportion of the split not used for training is to be used for 
            validating/testing? >0.5: more data for validation; <0.5: more data for testing. Default: 0.5.
        save_datasplit: Boolean flag whether to save the applied data split as W&B artifact. Default: True.
        save_overview: Boolean flag whether to save the idx/age overview as W&B artifact. 
            Only effective with save_datasplit=True. Default: False.
        all_data: boolean flag to indicate whether to use all data or only a subset of 100 samples. Default: True.
        test: boolean flag whether to test the trained model's performance on the test set. Default: False.
        finish: boolean flag whether to finish a wandb run. Default: False.
        execution: whether model is called from a Jupyter Notebook ('nb') or the terminal ('t'). 
                    Teriminal call cannot handle dilation. Default: 'nb'.
    Output: (if finish=False)
        trainer: trainer instance (after model training).
        datamodule: datamodule used for training model.
    """
    if name is None:
        name = config['name']
    if tags is None:
        tags = config['tags']
    if max_epochs is not None:
        config['parameters']['max_epochs'] = int(max_epochs)
        
    # start wandb run
    with wandb.init(project=config['project'],
                    group=config['group'],
                    name=name,
                    tags=tags,
                    config=config['parameters']) as run: 
        # update config with additional settings
        run.config['batch_size'] = batch_size
        
        run.config['dev'] = dev
        run.config['seed'] = seed
        run.config['train_ratio'] = train_ratio
        run.config['val_test_ratio'] = val_test_ratio
        run.config['all_data'] = all_data
        if use_gpu:
            run.config['accelerator'] = 'gpu'
            if devices is None:
                devices = [1]
        else:
            # make sure only one CPU thread is used
            torch.set_num_threads(num_threads)
            run.config['accelerator'] = 'cpu'
            devices = 'auto'
        run.config['lr_scheduler_config_path'] = lr_scheduler_config_path
            
        updated_config = run.config

        # increase reproducibility
        utils.make_reproducible(updated_config.seed)
        
        # initialise logger
        wandb_logger = WandbLogger(log_model='all')
        
        # initialise callbacks
        checkpoint = checkpoint_init()
        early_stopping = utils.earlystopping_init(patience=updated_config.patience)
        callbacks = [checkpoint, early_stopping]
        # add learning rate monitor if lr is variable
        if lr_scheduler_config_path is not None:
            lr_monitor = LearningRateMonitor() #logging_interval='epoch'
            callbacks += [lr_monitor]

        # initialise trainer
        trainer = pl.Trainer(accelerator=updated_config.accelerator, 
                             devices=devices,
                             logger=wandb_logger,
                             log_every_n_steps=updated_config.log_steps,
                             max_epochs=updated_config.max_epochs,
                             callbacks=callbacks,
                             deterministic=True)
        
        # depending on dataset modality
        if run.group == 'schaefer_ts':
            # initialise DataModule
            datamodule = data.UKBBDataModule(data_path='/ritter/share/data/UKBB/ukb_data/',
                                             dataset_type='UKBB_Schaefer_ts',
                                             schaefer_variant=updated_config.schaefer_variant,
                                             corr_matrix=updated_config.corr_matrix,
                                             shared_variants=updated_config.shared_variants,
                                             additional_data_path=updated_config.additional_data_path,
                                             heldout_set_name=updated_config.heldout_set_name,
                                             all_data=updated_config.all_data,
                                             dev=updated_config.dev,
                                             batch_size=updated_config.batch_size, 
                                             seed=updated_config.seed, 
                                             train_ratio=updated_config.train_ratio, 
                                             val_test_ratio=updated_config.val_test_ratio,
                                            )
        
            # initialise variable1DCNN model
            model = models.variable1DCNN(in_channels=updated_config.in_channels,
                                        kernel_size=updated_config.kernel_size,
                                        lr=updated_config.lr,
                                        depth=updated_config.depth,
                                        start_out=updated_config.start_out,
                                        scale_dim=updated_config.scale_dim,
                                        stride=updated_config.stride,
                                        weight_decay=updated_config.weight_decay,
                                        conv_dropout=updated_config.conv_dropout,
                                        final_dropout=updated_config.final_dropout,
                                        double_conv=updated_config.double_conv,
                                        batch_norm=updated_config.batch_norm,
                                        execution=execution,
                                        lr_scheduler_config_path=updated_config.lr_scheduler_config_path) 

            # model = models.debuggingCNN(in_channels=updated_config.in_channels,
            #                             kernel_size=updated_config.kernel_size,
            #                             lr=updated_config.lr,
            #                             depth=updated_config.depth,
            #                             start_out=updated_config.start_out,
            #                             scale_dim=updated_config.scale_dim,
            #                             stride=updated_config.stride,
            #                             weight_decay=updated_config.weight_decay,
            #                             conv_dropout=updated_config.conv_dropout,
            #                             final_dropout=updated_config.final_dropout,
            #                             double_conv=updated_config.double_conv,
            #                             batch_norm=updated_config.batch_norm,
            #                             execution=execution,
            #                             lr_scheduler_config_path=updated_config.lr_scheduler_config_path) 

        # train model
        trainer.fit(model, datamodule=datamodule)
        
        if save_datasplit:
            # take note of applied data split
            # create a local data_info directory (with or without overview)
            if save_overview:
                utils.save_data_info('', datamodule, only_indices=False)
            else:
                utils.save_data_info('', datamodule, only_indices=True)
            # create artifact for data split
            split_artifact = wandb.Artifact(name='split_indices', type='dataset')
            split_artifact.add_dir('data_info')
            # finalise artifact
            run.finish_artifact(split_artifact)
            # remove local data_info directory
            shutil.rmtree('data_info')
        
        # test model
        if test:
            trainer.test(ckpt_path='best', datamodule=datamodule)

        # R2
        y_pred_val = trainer.predict(model, 
                                     dataloaders=datamodule.general_dataloader(datamodule.data,datamodule.predict_sampler_val),
                                     ckpt_path='best')
        y_pred_test = trainer.predict(model, 
                                      dataloaders=datamodule.general_dataloader(datamodule.data,datamodule.predict_sampler_test),
                                      ckpt_path='best')
        val_ids_list, y_pred_val_list = utils.collect_predictions(y_pred_val)
        test_ids_list, y_pred_test_list = utils.collect_predictions(y_pred_test)

        run.summary['val_r2'] = r2_score(utils.get_true_ys(datamodule.data.labels,val_ids_list), y_pred_val_list)
        run.summary['test_r2'] = r2_score(utils.get_true_ys(datamodule.data.labels,test_ids_list), y_pred_test_list)
        
        if not finish:
            # return trainer instance + datamodule
            return trainer, datamodule
        
    if finish:
        # finish run
        wandb.finish()

# testing a model
def wandb_test(trainer, datamodule, finish=True):
    """
    Evaluates a trained model on the test set.
    Input:
        trainer: trainer instance (after model training).
        datamodule: datamodule used for training model.
        finish: boolean flag whether to finish a wandb run. Default: True.
    """
    trainer.test(ckpt_path='best', datamodule=datamodule)
    
    if finish:
        # finish run
        wandb.finish()
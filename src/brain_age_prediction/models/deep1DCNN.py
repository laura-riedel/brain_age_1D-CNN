# PyTorch modules
import torch
from torch import optim, nn
import pytorch_lightning as pl

# own utils module
from brain_age_prediction import utils

################################################################################################

# define LightningModule
class deep1DCNN(pl.LightningModule):
    """
    1D Convolutional Neural Network (CNN) that takes (fMRI) timeseries as input and predicts
    (i.e., regresses) a participant's age.
    Explicit, non-repetitive version of the final deep variable1DCNN that was determined 
    through a series of experiments.
    For use with SHAP, this architecture uses no nn.Flatten() or repeating layers.
    Input:
        loss: loss to be used.
        lr: learning rate to be used.
        weight_decay: weight decay for optimiser.
        lr_scheduler_config_path: path to learning rate scheduler configuration file. 
                    If present, sets a learning rate scheduler. Ignored if None. Default: None.
    Output:
        A model.
    """
    def __init__(self, loss=nn.MSELoss(), lr=1e-3, weight_decay=0, 
                 lr_scheduler_config_path=None):
        super().__init__()
        self.loss = loss
        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_scheduler_config_path = lr_scheduler_config_path
        # save hyperparameters
        self.save_hyperparameters()
        # make reproducible
        utils.make_reproducible()        
        
        # additional variables
        # for tracking best validation loss during training
        self.best_val_loss = 10000
        # tracking validation step outputs
        self.validation_step_outputs = []
                
        # define model architecture
        # encoder
        self.encoder = nn.Sequential(
            # layer 1
            nn.Conv1d(in_channels=100, out_channels=128, kernel_size=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3,stride=2),
            # layer 2
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3,stride=2),
            # layer 3
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3,stride=2),
            # layer 4
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(in_features=128, out_features=1)
        )
    
    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions          
        x = self.encoder(x)
        # flatten
        x = x.view(-1, 128)
        return self.decoder(x)
        
    def training_step(self, batch, batch_idx): 
        # training_step defines the train loop, independent of forward
        x, y, _ = batch
        y_hat = self.forward(x)
        y = torch.unsqueeze(y,1)
        loss = self.loss(y_hat, y)
        
        self.log('train_loss', loss) #, on_step=True, on_epoch=True, logger=True
        return loss
    
    def predict_step(self, batch, batch_idx):
        # step function called during predict()
        with torch.no_grad():
            x, y, sub_id = batch
            y_hat = self.forward(x)
            return sub_id, y_hat
    
    def evaluate(self, batch, stage=None):
        """Helper function that generalises the steps for validation_step and test_step.
        Calculates loss & MAE and logs both values.
        Input:
            batch: current batch.
            stage: 'val' for validation_step or 'test' for test_step.
        """
        x, y, _ = batch
        y_hat = self.forward(x)
        y = torch.unsqueeze(y,1)
        loss = self.loss(y_hat, y)
        mae = nn.functional.l1_loss(y_hat, y)
        
        if stage:
            self.log(f'{stage}_loss', loss) #, on_step=True, on_epoch=True, logger=True
            self.log(f'{stage}_mae', mae) #, on_step=True, on_epoch=True, logger=True
            
        if stage == 'val':
            # track validation losses in list
            self.validation_step_outputs.append(loss)
            return {"val_loss": loss, "diff": (y - y_hat), "target": y, 'mae': mae}
    
    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, 'val')
        
    def on_validation_epoch_end(self):
        #val_mean = torch.cat([x['val_loss'] for x in outputs]).mean(dim=0)
        val_mean = torch.stack(self.validation_step_outputs).mean()
        # update best val loss if current loss is smaller
        if val_mean < self.best_val_loss:
            self.best_val_loss = val_mean
        # log current best val loss
        self.log('best_val_loss', self.best_val_loss)
        # clear memory
        self.validation_step_outputs.clear()
    
    def test_step(self, batch, batch_idx):
        self.evaluate(batch, 'test')
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay) 
        if self.lr_scheduler_config_path is None:
            return optimizer
        else:
            return {'optimizer': optimizer, 
                    'lr_scheduler': utils.load_lr_scheduler_config(self.lr_scheduler_config_path, optimizer)}

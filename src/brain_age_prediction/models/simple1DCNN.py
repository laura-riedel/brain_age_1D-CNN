# PyTorch modules
import torch
from torch import optim, nn
import pytorch_lightning as pl

# own utils module
from brain_age_prediction import utils

################################################################################################

# define LightningModule
class simple1DCNN(pl.LightningModule):
    """
    Simple 1D Convolutional Neural Network (CNN) that takes (fMRI) timeseries as input and predicts
    (i.e., regresses) a participant's age.
    Input:
        channels: number of components/regions/input channels.
        activation: activation function to be used.
        loss: loss to be used.
        lr: learning rate to be used.
    Output:
        A trained model.
    """
    def __init__(self, in_channels=25, activation=nn.ReLU(), loss=nn.MSELoss(), lr=1e-3):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = 5
        self.act = activation
        self.loss = loss
        self.lr = lr
        self.save_hyperparameters()
        utils.make_reproducible()        
        
        # define convolutional and maxpool layers
        self.conv1 = nn.Conv1d(self.in_channels, 32, kernel_size=self.kernel_size)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=self.kernel_size)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=self.kernel_size)
        self.conv4 = nn.Conv1d(128, 256, kernel_size=self.kernel_size)
        self.maxpool = nn.MaxPool1d(kernel_size=self.kernel_size, stride=2)
                
        # define model architecture
        self.model = nn.Sequential(self.conv1,
                                    self.act,
                                    self.maxpool,
                                    self.conv2,
                                    self.act,
                                    self.maxpool,
                                    self.conv3,
                                    self.act,
                                    self.maxpool,
                                    self.conv4,
                                    self.act,
                                    self.maxpool,
                                    nn.Flatten(),
                                    nn.Linear(6144,1)
                                    )
    
    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        return self.model(x)
        
    def training_step(self, batch, batch_idx): 
        # training_step defines the train loop, independent of forward
        x, y, _ = batch
        y_hat = self.forward(x)
        y = torch.unsqueeze(y,1)
        loss = self.loss(y_hat, y)
        
        self.log('train_loss', loss)
        return loss
    
    def evaluate(self, batch, stage=None):
        x, y, _ = batch
        y_hat = self.forward(x)
        y = torch.unsqueeze(y,1)
        loss = self.loss(y_hat, y)
        mae = nn.functional.l1_loss(y_hat, y)
        
        if stage:
            self.log(f'{stage}_loss', loss)
            self.log(f'{stage}_mae', mae)
            
        if stage == 'val':
            return {"val_loss": loss, "diff": (y - y_hat), "target": y, 'mae': mae}
    
    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, 'val')
    
    def test_step(self, batch, batch_idx):
        self.evaluate(batch, 'test')
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr) 
        return optimizer

# PyTorch modules
import torch
from torch import optim, nn
import pytorch_lightning as pl

# own utils module
from brain_age_prediction import utils

################################################################################################

# define LightningModule
class debuggingCNN(pl.LightningModule):
    """
    Debugging the variable1DCNN
    """
    def __init__(self, in_channels=25, kernel_size=5, activation=nn.ReLU(), loss=nn.MSELoss(), 
                 lr=1e-3, depth=4, start_out=32, scale_dim=True, stride=2, weight_decay=0, dilation=1,
                 conv_dropout=0, final_dropout=0, double_conv=False, batch_norm=False, execution='nb', 
                 lr_scheduler_config_path=None, 
                 flatten=True):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.act = activation
        self.loss = loss
        self.lr = lr
        self.depth = depth
        self.start_out = start_out
        self.scale_dim = scale_dim
        self.stride = stride
        self.conv_dropout_p = conv_dropout
        self.final_dropout_p = final_dropout
        self.weight_decay = weight_decay
        self.dilation = dilation
        self.double_conv = double_conv
        self.batch_norm = batch_norm
        self.execution = execution
        self.flatten = flatten
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
        
        # define maxpool layer
        self.maxpool = nn.MaxPool1d(kernel_size=self.kernel_size, stride=self.stride)
                
        # define model architecture
        # encoder
        channel = self.start_out
        self.encoder = nn.Sequential()
        # add X sequences of convolutional layer, activation + maxpool
        for layer in range(self.depth):
            # treat first layer a bit differently
            if layer == 0:
                self.add_conv_layer(self.in_channels, channel)
            # all other layers
            else:
                if scale_dim:
                    new_channel = channel*2
                else:
                    new_channel = channel
                self.add_conv_layer(channel, new_channel)
                # set output channel size as new input channel size
                channel = new_channel
            # add second conv layer if flag = True (complete with activation (+ BatchNorm + dropout))
            if self.double_conv:
                if scale_dim:
                    new_channel = channel*2
                else:
                    new_channel = channel
                self.add_conv_layer(channel, new_channel)
                # update output channel size again as new input channel size
                channel = new_channel
            # add pooling layer
            if layer == self.depth-1:
                # average pooling after last conv layer
                # self.encoder.append(nn.AvgPool1d(self.kernel_size, self.stride))
                self.encoder.append(nn.AdaptiveAvgPool1d(1))
            else:
                # normally maxpool
                self.encoder.append(self.maxpool)
        # add a final flatten layer at the end
        if self.flatten:
            print('With flatten:')
            self.encoder.append(nn.Flatten())
        else:
            print('Without flatten (AvgPool output):')
        
        
        # check: too many parameters?
        total_encoder_params = self.get_num_parameters(self.encoder)   
        # threshold: 5,000,000
        if total_encoder_params > 5000000:
            raise Exception(f'More than 5Mio parameters in encoder! Count: {total_encoder_params}')
        
        # decoder
        flattened_dimension = self.get_flattened_dimension(self.encoder)
        self.decoder = nn.Sequential()
        # add dropout if wanted
        if self.final_dropout_p:
            self.decoder.append(nn.Dropout1d(p=self.final_dropout_p))
        self.decoder.append(nn.Linear(flattened_dimension, 1))
        
    def add_conv_layer(self, in_channel, out_channel):
        """ Function to add a convolutional layer and activation function
        (and optionally BatchNorm and conv dropout) to encoder architecture.
        Input:
            in_channel: input channel dimension.
            out_channel: output channel dimension.
        """
        if self.execution == 'nb':
            # Jupyter Notebook can handle dilation
            conv_layer = nn.Conv1d(in_channel, out_channel, kernel_size=self.kernel_size, dilation=self.dilation)
        else: 
            # terminal cannot handle dilation
            conv_layer = nn.Conv1d(in_channel, out_channel, kernel_size=self.kernel_size)
        # append to list
        self.encoder.append(conv_layer)
        
        # add BatchNorm if flag = True
        if self.batch_norm:
            self.encoder.append(nn.BatchNorm1d(out_channel))
            
        # add activation to encoder
        self.encoder.append(self.act)
        
        # add dropout if wanted
        if self.conv_dropout_p:
            self.encoder.append(nn.Dropout1d(p=self.conv_dropout_p))

    def get_flattened_dimension(self, model):
        """Function to figure out the dimension after nn.Flatten()
        so that it can be fed into the linear layer.
        Takes a given model, creates dummy input, and checks the
        dimensions after the input has gone through the model.
        Input:
            model: a (sequential) model architecture
        Output:
            input dimension size for linear layer
        """
        input = torch.randn(10, self.in_channels, 490)
        x = model(input)
        print("output get_flattened_dimension:", x.size(1))
        return x.size(1)
    
    def get_num_parameters(self, model):
        """
        Calculate how many parameters a model uses overall.
        """
        return sum(torch.numel(p) for p in model.parameters())
    
    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions          
        x = self.encoder(x)
        print('Shape after encoder:',x.shape)
        x = self.decoder(x)
        print('Shape after decoder',x.shape)
        return x
        
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

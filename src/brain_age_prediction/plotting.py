import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

#### VISUALISATIONS
def plot_training(data, yscale='log', title='', xmin=None, xmax=None):
    """
    Plots the training process of a model. 
    Expects the metrics_df from get_metrics as input.
    """
    if xmin == None:
        xmin = data.index.min()
    if xmax == None:
        xmax = data.index.max()
    # only include train + validation values in plot
    visualisation = sns.relplot(data=data.loc[xmin:xmax,'train_loss':'val_mae'], kind='line')
    # set scale and title
    visualisation.set(yscale=yscale)
    plt.title(title)
    plt.show()

def get_plot_values(df):
    """
    Get values to make nice plotting of final model training easier.
    """
    xmin = df.index.min()
    xmax = df.index.max()
    xstep = int(np.floor(xmax/10))
    xrange = [i for i in range(xmin, xmax, xstep)]
    return xmin, xmax, xstep, xrange
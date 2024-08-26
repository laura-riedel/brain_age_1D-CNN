import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

#### VISUALISATIONS
# TRAINING 
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

# BAGs
def preds_corr_overview(df, variables=True, models=None):
    """
    Calculate correlations between all aspects of interest.
    Expects an overview dataframe that is limited to those IDs for which
    predictions exist.
    Input:
        df: (heldout) overview pandas dataframe.
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
        correlations_df: correlation overview dataframe.
    """
    var_columns = ['bmi', 'digit substitution', 'education', 'fluid intelligence',
                   'grip', 'depressive episode', 'all depression',
                   'recurrent depressive disorder', 'multiple sclerosis', 'sex',
                   'weekly beer', 'genetic pc 1', 'genetic pc 2', 'genetic pc 3']
    bag = 'bag_'
    connector = '_'
    if not models:
        models = ['']
        bag = 'bag'
        connector = ''
    if variables:
        correlations_df = pd.DataFrame(columns=['Variable'])
        idx = 0
        for column in var_columns:
            for model in models:
                correlations_df.loc[idx,'Variable'] = column
                correlations_df.loc[idx,'Corr BAG '+model+' model'] = df[bag+model].corr(df[column], method='spearman')
                correlations_df.loc[idx,'Corr detrended BAG '+model+' model'] = df[bag+model+'_detrended'].corr(df[column], method='spearman')
            idx += 1
    else:
        rows = []
        corr_cols = []
        for model in models:
            rows.append(f'Predicted age {model} model')
            rows.append(f'BAG {model} model')
            rows.append(f'Detrended BAG {model} model')
            corr_cols.append(f'predicted_age{connector}{model}')
            corr_cols.append(f'bag{connector}{model}')
            corr_cols.append(f'bag{connector}{model}_detrended')
        correlations_df = pd.DataFrame(columns=['True age vs.','Corr'])
        for idx in range(len(rows)):
            correlations_df.loc[idx,'True age vs.'] = rows[idx]
            correlations_df.loc[idx,'Corr'] = df['age'].corr(df[corr_cols[idx]], method='spearman')
    return correlations_df

def bootstrap_overview(corrs_dict, variables=True, model=None):
    var_columns = ['bmi', 'digit substitution', 'education', 'fluid intelligence',
                   'grip', 'depressive episode', 'all depression',
                   'recurrent depressive disorder', 'multiple sclerosis', 'sex',
                   'weekly beer', 'genetic pc 1', 'genetic pc 2', 'genetic pc 3']
    connector = '_'
    if not model:
        model = ''
        connector = ''
    if variables:
        correlations_df = pd.DataFrame(columns=['Variable'])
        for idx in range(len(var_columns)):
            correlations_df.loc[idx,'Variable'] = var_columns[idx]
            correlations_df.loc[idx,'Corr BAG '+model+' model mean'] = corrs_dict['corrs variables']['mean'][idx]
            correlations_df.loc[idx,'Corr BAG '+model+' model sem'] = corrs_dict['corrs variables']['std'][idx]
            correlations_df.loc[idx,'Corr detrended BAG '+model+' model mean'] = corrs_dict['corrs variables detrended']['mean'][idx]
            correlations_df.loc[idx,'Corr detrended BAG '+model+' model sem'] = corrs_dict['corrs variables detrended']['std'][idx]
    else:
        rows = []
        corr_cols = []
        rows.append(f'Predicted age {model} model')
        rows.append(f'BAG {model} model')
        rows.append(f'Detrended BAG {model} model')
        corr_cols.append(f'predicted_age{connector}{model}')
        corr_cols.append(f'bag{connector}{model}')
        corr_cols.append(f'bag{connector}{model}_detrended')
        correlations_df = pd.DataFrame(columns=['True age vs.','Corr mean','Corr sem'])
        for idx in range(len(rows)):
            correlations_df.loc[idx,'True age vs.'] = rows[idx]
            correlations_df.loc[idx,'Corr mean'] = corrs_dict['corrs true age']['mean'][idx]
            correlations_df.loc[idx,'Corr sem'] = corrs_dict['corrs true age']['std'][idx]
    return correlations_df

def bag_viz(df, variable_name, plot_type, label_x=None, detrended=True, y_ticks=[-25,-20,-15,-10,-5,0,5,10,15,20,25], fig_path=None):
    """
    Create (and optionally save) figures for the correlation
    of BAGs to a select health or demographic variable.
    Input:
        df: heldout overview df with BAG (+ detrended BAG) info.
        variable_name: name of the variable column of interest.
        plot_type: determines how to plot; one of: 
            'regplot', 'boxplot', or 'pointplot'.
        label_x: how to display the variable name as x-axis label.
            Defaults to variable name.
        detrended: Boolean flag determining whether to use the initial
            or linearly detrended BAG values. Default: True.
        y_ticks: sets yticks of both subplots in given range.
        fig_path: path + name to save the figure to, if provided. 
            Default: None.
    """
    orig_y_name = 'bag_orig'
    new_y_name = 'bag_new'
    if detrended:
        orig_y_name += '_detrended'
        new_y_name += '_detrended'
    if label_x is None:
        label_x = variable_name
    fig, axes = plt.subplots(1,2,figsize=(15, 5))
    # bag plots
    if plot_type=='regplot':
        sns.regplot(data=df, y=orig_y_name, x=variable_name, ax=axes[0], line_kws=dict(color='#ff7f0e'))
        sns.regplot(data=df, y=new_y_name, x=variable_name, ax=axes[1], line_kws=dict(color='#ff7f0e'))
    elif plot_type=='boxplot':
        sns.boxplot(data=df, y=orig_y_name, x=variable_name, ax=axes[0])
        sns.boxplot(data=df, y=new_y_name, x=variable_name, ax=axes[1])
    elif plot_type=='pointplot':
        sns.pointplot(data=df, y=orig_y_name, x=variable_name, ax=axes[0])
        sns.pointplot(data=df, y=new_y_name, x=variable_name, ax=axes[1])
    else:
        raise Exception("No valid plot type defined.")    
    # ax configs + line at y=0 in background
    for ax in axes:
        ax.axhline(y=0, color='#7f7f7f', linestyle='--', zorder=0)
        ax.set(ylabel='Brain age gap',
            yticks=y_ticks, 
            xlabel=label_x,
            )
    axes[0].set(title='Original model')
    axes[1].set(title='New model')
    if fig_path:
        plt.savefig(fig_path, bbox_inches='tight')
    fig.show()

# EXPLAINABILITY
def get_network_mapping():
    network_mapping = {
        'Vis': 'visual',
        'SomMot': 'somatomotor',
        'DorsAttn': 'dorsal attention',
        'SalVentAttn': 'salience / ventral attention',
        'Limbic': 'limbic',
        'Cont': 'control',
        'Default': 'default',
    }
    return network_mapping

def get_area_mapping():
    area_mapping = {
        'Post': 'posterior',
        'PrCv': 'precentral ventral',
        'FEF': 'frontal eye fields',
        'ParOper': 'parietal operculum',
        'FrOperIns': 'frontal operculum insula',
        'PFCl': 'prefrontal cortex',
        'PFCl SalVentAttn': 'prefrontal cortex',
        'PFCl Cont': 'prefrontal cortex',
        'Med': 'medial',
        'OFC': 'orbitofrontal cortex',
        'TempPole': 'temporal pole',
        'Par': 'parietal',
        'Par Cont': 'parietal',
        'Par Default': 'parietal',
        'pCun': 'precuneus',
        'Cing': 'cingulate',
        'Temp': 'temporal',
        'PFC': 'prefrontal cortex',
        'pCunPCC': 'precuneus posterior cingulate cortex',
        'TempOccPar': 'temporal occipital parietal',
        'PFCmp': 'prefrontal cortex',
        'PFCv': 'prefrontal cortex',
        'PFCdPFCm': 'prefrontal cortex',
    }
    return area_mapping

def get_colour_map(map_type='network'):
    """
    Load dictionary with specific colour style.
    Input:
        map_type: either 'network' or 'area'.
    Output:
        colour_map: colour map dictionary mappying either networks or areas to colours.
    """
    if map_type == 'network':
        colour_map = {
            'Vis': '#E7FFA6',
            'SomMot': '#FFE7A6',
            'DorsAttn': '#FFA6BE',
            'SalVentAttn': '#DEA6FF',
            'Limbic': '#A6BBFF',
            'Cont': '#A6FFFE',
            'Default': '#A6FFB2',
        }
    elif map_type == 'area':
        colour_map = {
            # visual
            'full Vis': '#E7FFA6',
            # somatomotor
            'full SomMot': '#FFE7A6',
            # salience / ventral attention
            'Post': '#FFA6BE',
            'PrCv': '#E57E9A',
            'FEF': '#CC5C7A',
            'ParOper': '#DEA6FF',
            'TempOccPar': '#C07EE5',
            'FrOperIns': '#A35CCC',
            'PFCl SalVentAttn': '#883EB3',
            'Med': '#6F2699',
            # limbic
            'OFC': '#A6BBFF',
            'TempPole': '#7E96E5',
            # control
            'Par Cont': '#A6FFFE',
            'PFCl Cont': '#7EE5E4',
            'pCun': '#5CCCCA',
            'Cing': '#3EB3B1',
            'PFCmp': '#7EE5E4',
            # default        
            'Temp': '#A6FFB2',
            'Par Default': '#7EE58C',
            'PFC': '#5CCC6B',
            'pCunPCC': '#3EB34E',
            'PFCv': '#5CCC6B',
            'PFCdPFCm': '#5CCC6B',
        }
    else:
        raise NameError('map_type must be "network" or "area". Check for spelling errors!')
    return colour_map
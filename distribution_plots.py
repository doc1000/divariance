import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from divariance import Empirical, Formulas
plt.style.use('fivethirtyeight')
plt.style.use('ggplot')

def generate_distributions(mean1, mean2, var1, var2, corr, size=1000):
    """
    Generate two distributions with specified means, variances, and covariance.

    Parameters:
        mean1 (float): Mean of the first distribution.
        mean2 (float): Mean of the second distribution.
        var1 (float): Variance of the first distribution.
        var2 (float): Variance of the second distribution.
        #cov (float): Covariance between the two distributions.
        corr (float): correlation between the two distributions
        size (int): Number of samples to generate.

    Returns:
        x (np.ndarray): Samples from the first distribution.
        y (np.ndarray): Samples from the second distribution.
    """
    # calculate covar
    cov = corr*np.sqrt(var1)*np.sqrt(var2)
    
    # Create the covariance matrix
    cov_matrix = np.array([[var1, cov],
                           [cov, var2]])
    # Mean vector
    mean_vector = np.array([mean1, mean2])
    # Generate samples
    samples = np.random.multivariate_normal(mean_vector, cov_matrix, size)
    x, y = samples[:, 0], samples[:, 1]
    return x, y


distr_default = {'mean1':0, 'mean2':0, 'var1':1, 'var2':1, 'corr':1, 'size':1000}

def generate_metrics(metric, arange, distr_default):
    distr_inputs = distr_default.copy()
    dirr, corr,divar, prod_sig = [],[], [],[]
    kl_xy, kl_yx = [],[]
    x_s, y_s = [],[]
    fre_con, kl_f, rdd = [], [], []
    for value in arange:
        distr_inputs[metric]=value
        x, y = generate_distributions(**distr_inputs)
        x_s.append(x)
        y_s.append(y)
        dirr.append(Empirical.dirrelation(x,y))
        corr.append(np.corrcoef(x,y)[0,1])
        kl_xy.append(Empirical.kl_divergence(x,y))
        kl_yx.append(Empirical.kl_divergence(y,x))
        kl_f.append(Formulas.kl_normal(**distr_inputs,return_mean=True))
        rdd.append(Empirical.root_dirrelation_divergence(x,y))
        fre_con.append(Formulas.frechet_divergence(**distr_inputs))
        divar.append(Empirical.divariance(x,y))
        prod_sig.append(np.std(x)*np.std(y))

    kl_mean = 0.5*(np.array(kl_yx)+np.array(kl_xy))

    metrics = {'dirrelation':dirr,'root dirrelation divergence': rdd,
            'kl_PQ':kl_xy,'kl_QP':kl_yx,'kl_mean':kl_mean,
              'correlation':corr,'dicorrelation':np.array(corr)*np.array(dirr)
              ,'kl_formula':kl_f, 'x_s':x_s,'y_s':y_s, 'frechet divergence':fre_con
              ,'divariance':divar, 'sigma1*sigma2': prod_sig}

    #return dirr, corr, x_s, y_s, kl_xy, kl_yx, dirr_f, kl_f
    return metrics


def plot_density(x,y, ax=None,return_ax=True,figsize=None,xlabel='x',ylabel='y'):
    if ax is None:
        if figsize is None:
            fig, ax = plt.subplots()
        else:
            fig, ax = plt.subplots(figsize=figsize)
    sns.set_style('whitegrid')
    sns.kdeplot(x, bw_method=0.5,label=xlabel,ax=ax)
    sns.kdeplot(y, bw_method=0.5,label=ylabel,ax=ax)
    #ax.set_title('KDE Plot of Generated Distributions')
    ax.legend();
    if return_ax:
        return ax;
    else:
        plt.show();

def plot_scatter(x,y, ax=None,return_ax=True,show_ylabel=True):
    if ax is None:
        fig, ax = plt.subplots()
    ax.scatter(x, y, alpha=0.5)
    ax.set_xlabel('Distr x')
    if show_ylabel:
        ax.set_ylabel('Distr y')
        #ax.set_title('Scatter Plot of Generated Distributions')
    if return_ax:
        return ax
    else:
        plt.show()

def plot_metrics_packed(y_data: dict,arange,xlabel='Mean changes: mean2 value',ax=None,return_ax=True):
    if ax is None:
        fig, ax = plt.subplots()
    for i,k in enumerate(y_data.keys()):
        v = y_data[k]
        if i <3:
            linestyle='solid'
        else: 
            linestyle = ':'
        ax.plot(arange,v,label=k,alpha=0.7,linestyle=linestyle)
    ax.legend()
    if return_ax:
        return ax


def plot_metric_subplots(metric,arange,plot_keys,distr_default,scale = 'base',title=None ,
                         savefile=None,include_density=True,include_scatter=True,ylabel='Divergence'):
    metrics = generate_metrics(metric, arange, distr_default)
    x_s, y_s = metrics['x_s'],metrics['y_s']
    y_data = {k: metrics[k] for k in plot_keys}
    #fig, ax = plt.subplots(figsize=[10,4])
    #plt.style.use('default')
    plt.style.use('seaborn-v0_8-whitegrid')
    fig = plt.figure(figsize=[10,8])
    ax = plt.subplot(2,1,1)
    #ax.patch.set_facecolor('white')
    ax.set_ylabel(ylabel)
    #ax.set_xlabel(metric)
    if scale == 'log':
        ax.set_xscale('log')
    ax = plot_metrics_packed(y_data,arange,xlabel=f'Metric changes: {metric}',ax=ax)
    
    if title is None:
        default_plot_title = "Effect on KL Divergence and Root Dirrelation Divergence\nChanging Metric: "
        title = f"{default_plot_title}{metric}"
    ax.set_title(title)
    midpoint = int(len(x_s)/2)
    if ('correlation' in plot_keys) & ('dicorrelation' in plot_keys):
        ax.annotate("",xytext=(arange[0],corr[0]),xy=(arange[0],metrics['dicorrelation'][0]),arrowprops=dict(facecolor='black', shrink=0.05)) #,xycoords='data')
        #ax.annotate("",xytext=(arange[0],corr[0]),xy=(arange[0],(corr[0]/dirr[0])),arrowprops=dict(facecolor='black', shrink=0.05)) #,xycoords='data')
        ax.annotate("",xytext=(arange[-1],corr[-1]),xy=(arange[-1],metrics['dicorrelation'][-1]),arrowprops=dict(facecolor='black', shrink=0.05)) #,xycoords='data')
        #ax.annotate("",xytext=(arange[-1],corr[-1]),xy=(arange[-1],(corr[-1]/dirr[-1])),arrowprops=dict(facecolor='black', shrink=0.05)) #,xycoords='data')
        
        #ax.annotate("",xytext=(arange[midpoint],corr[midpoint]),xy=(arange[midpoint],(corr[midpoint]/dirr[midpoint])),arrowprops=dict(facecolor='black', shrink=0.05)) #,xycoords='data')

    axes=[]
    height_max = 4
    starting_position=7
    if include_density:
        
        #fig, axes = plt.subplots(1,3,figsize=[10,1.5],sharex=True,sharey=True)
        #for x in np.arange(7,13):
        #    ax = plt.subplot(height_max,3,x)
        #    axes.append(ax)
        axes.append(plt.subplot(height_max,3,starting_position))
        axes.append(plt.subplot(height_max,3,starting_position+1,sharex=axes[0],sharey=axes[0]))
        axes.append(plt.subplot(height_max,3,starting_position+2,sharex=axes[0],sharey=axes[0]))
        starting_position+=3
        _ = plot_density(x_s[0],y_s[0], ax=axes[0],return_ax=True)
        _ = plot_density(x_s[midpoint],y_s[midpoint], ax=axes[1],return_ax=True)
        _ = plot_density(x_s[-1],y_s[-1], ax=axes[2],return_ax=True)
        axes[0].set_title(f"{metric}={arange[0]:.2f}",size='medium')
        axes[1].set_title(f"{metric}={arange[midpoint]:.2f}",size='medium')
        axes[2].set_title(f"{metric}={arange[-1]:.2f}",size='medium')
        #fig.suptitle(f"Density plots at lowest, middle and highest {metric} value")
        #fig, axes = plt.subplots(1,3,figsize=[10,1.5],sharex=True,sharey=True)
    if include_scatter:
        axes.append(plt.subplot(height_max,3,starting_position))
        _ = plot_scatter(x_s[0],y_s[0], ax=axes[-1],return_ax=True)

        axes.append(plt.subplot(height_max,3,starting_position+1,sharex=axes[-1],sharey=axes[-1]))
        _ = plot_scatter(x_s[midpoint],y_s[midpoint], ax=axes[-1],return_ax=True,show_ylabel=False)

        axes.append(plt.subplot(height_max,3,starting_position+2,sharex=axes[-1],sharey=axes[-1]))        
        _ = plot_scatter(x_s[-1],y_s[-1], ax=axes[-1],return_ax=True,show_ylabel=False)
        
    plt.tight_layout(h_pad=0.3,w_pad=1)
    if savefile is not None:
        plt.savefig(savefile,bbox_inches='tight')

    plt.show()

if __name__ == '__main__':
    print("Examples")
    distr_default = {'mean1':0, 'mean2':0.0, 'var1':1, 'var2':1, 'corr':1, 'size':1000}
    metric = 'var2'
    arange = np.exp(np.arange(np.log(0.1),np.log(10.1),0.1)) #log based range for equal iterations on each side of 1, good for variance, which is symmetrical by ratio
    distr_target=distr_default.copy()
    plot_keys = ['root dirrelation divergence','kl_mean','kl_PQ','kl_QP']
    plot_metric_subplots(metric,arange,plot_keys,distr_target,scale = 'log',title=None)#,savefile='kl_var.png')

    metric = 'mean2'
    arange = np.arange(-2.2,2.3,0.1)
    plot_metric_subplots(metric,arange,plot_keys,distr_target,scale = 'base',title=None)#,savefile='kl_mean.png')

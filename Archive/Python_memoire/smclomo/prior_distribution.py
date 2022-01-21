# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 17:22:08 2020

@author: pierr
"""
import pandas as pd
import numpy as np
import scipy.stats as st
import scipy.special as sp


def logp_gamma_prior(a, b):
    """
    log probabilities of independent gamma priors.

    Parameters
    ----------
    a : array 
        shape hyperparameters.
    b : array
        scale hyperparameters.
    

    Returns
    -------
    function
    Allows to evaluate the log probabilities in the proposed parameters.
    
    Example
    -------
    """
    def logp_prior(parms):
        if np.all(parms > 0):
            return(
                np.dot((a - 1), np.log(parms)) - np.dot(parms , b) + 
                np.dot(a , np.log(b)) - sum(np.log(sp.gamma( a )))
                   )
        else:
           return(-np.inf)
    return logp_prior

def sim_gamma_prior(a, b, parms_names, popSize):
    """
    Sample from independent gamma priors.

    Parameters
    ----------
    a : array 
        shape hyperparameters.
    b : array
        scale hyperparameters.
    parms_names: array
        name of the parameters
    popSize: int
        sample size
        
    Returns
    -------
    dataframe
    Initialize parameters value from the independent 
    gamma prior distribution.
    
    Example
    -------
    a, b, parms_name, popSize   = [0.1, 0.1, 0.1], [10, 10, 10],\
        ['k','α', 'θ'], 10
    sim_gamma_prior(a, b, parms_name, popSize)
    
    """
    mat = np.matrix([st.gamma(a[j]).rvs(popSize) / b[j] 
               for j in range(len(a))]).transpose()
    res = pd.DataFrame(mat)
    res.columns = parms_names
    return res

def logp_uniform_prior(a, b):
    """
    log probabilities of independent uniform priors.

    Parameters
    ----------
    a : array 
        lower bounds.
    b : array
        upper bounds.
    

    Returns
    -------
    function
    Allows to evaluate the log probabilities in the proposed parameters.
    
    Example
    -------
    """
    def logp_prior(parms):
        if np.all(parms > a) and np.all(parms < b) :
            return(-np.sum(np.log(b - a)))
        else:
            return(-np.inf)
    return logp_prior

def sim_uniform_prior(a, b, parms_names, popSize):
    """
    Sample from independent uniform priors.

    Parameters
    ----------
    a : array 
        shape hyperparameters.
    b : array
        scale hyperparameters.
    parms_names: array
        name of the parameters
    popSize: int
        sample size
        
    Returns
    -------
    dataframe
    Initialize parameters value from the independent 
    uniform prior distribution.
    
    Example
    -------
    a, b, parms_name, popSize   = [0, 0, 1], [10, 10, 260],\
        ['k','α', 'θ'], 10
    sim_uniform_prior(a, b, parms_name, popSize)
    
    """
    mat = np.matrix([st.uniform().rvs(popSize) * (b[j] - a[j]) + a[j] 
               for j in range(len(a))]).transpose()
    res = pd.DataFrame(mat)
    res.columns = parms_names
    return res


               

def logp_prior_wrap(model_prior, a, b):
    """
    Set the likelihood function for the chosen prior distribution.

    Parameters
    ----------
    model_prior: string
        name of the model
    a, b: float
        prior distribution hyper parameters

    Returns
    -------
    function
    Allows the evaluation of the prior log probabilities in the parameters .
    
    Example
    -------
    """
    if model_prior == "uniform":
        return(logp_uniform_prior(a, b))
    elif model_prior == "gamma":
        return(logp_gamma_prior(a, b))
    

def sim_prior_wrap(model_prior, a, b, parms_names, popSize):
    """
    Set the likelihood function for the chosen prior distribution.

    Parameters
    ----------
    model_prior: string
        name of the model
    a, b: float
        prior distribution hyper parameters
    parms_names: array
        names of the loss model parameters
    popSize: int
        number of particle in the cloud
    Returns
    -------
    function
    Allows the evaluation of the prior log probabilities in the parameters .
    
    Example
    -------
    """
    if model_prior == "uniform":
        return(sim_uniform_prior(a, b, parms_names, popSize))
    elif model_prior == "gamma":
        return(sim_gamma_prior(a, b, parms_names, popSize))
    
        

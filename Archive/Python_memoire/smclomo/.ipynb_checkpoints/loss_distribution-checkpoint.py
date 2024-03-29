# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 15:42:14 2020

@author: pierr
"""

import scipy.special as sp
import math as ma
import pandas as pd
import numpy as np
import scipy.stats as st
from scipy.optimize import minimize


def sim_gam_par(n, k, α, θ):
    """
    Sample from a Gamma-Pareto model.

    Parameters
    ----------
    n : int 
        sample size.
    k : float
        shape parameter of the Gamma distribution.
    α : float
        Tail index of the Pareto distribution.

    Returns
    -------
    array
    A sample drawn from the Weibull-Pareto distribution.
    
    Example
    -------
    k, α, θ = 1/2, 1/2, 5 
    X = sim_gam_par(1000, k, α, θ)
    """
    β = θ / (k + α)
    r = α*sp.gamma(k)*  sp.gammainc(k,  θ / β) * np.exp(k+α)*(k+α)**(-k) / \
    (1+ α*sp.gamma(k) * sp.gammainc(k, θ / β) * np.exp(k+α)*(k+α)**(-k))
    
    gamma_rv = st.gamma(k)
    par_rv = st.pareto(α)
    binom_rv =  st.binom(1, r)
    par_rvs = θ * par_rv.rvs(size = n)
    binom_rvs = binom_rv.rvs(size = n)
    gamma_rvs = β * gamma_rv.ppf(sp.gammainc(k, θ / β) *\
                               np.random.uniform(size = n))
    return(binom_rvs * gamma_rvs + (1 - binom_rvs) * par_rvs)

def logp_gam_par(X):
    """
    Likelihood function of the Gamma-Pareto model.

    Parameters
    ----------
    X : Array 
        Insurance losses.

    Returns
    -------
    function
    Allows the evaluation of the likelihood in the parameters provided the 
    data.
    
    Example
    -------
    k, α, θ = 1/2, 1/2, 5 
    X = sim_gam_par(100, k, α, θ)
    logp = logp_gam_par(X)
    logp(np.array([k, α, θ]))
    costFn = lambda parms: -logp(parms)
    bnds = ((0, None), (0, None), (0, None))
    θ0 = (1, 1, 1)
    minRes = minimize(costFn, θ0,bounds=bnds)
    minRes
    """
    # parms = particles.to_numpy()[4]
    def logp(parms):
        k, α, θ = tuple(parms)
        
        if np.all(parms > 0):
            β = θ / (k + α)
            r = α*sp.gamma(k)*  sp.gammainc(k,θ / β) * np.exp(k+α)*(k+α)**(-k) / \
            (1+ α*sp.gamma(k) * sp.gammainc(k, θ / β) * np.exp(k+α)*(k+α)**(-k))
            if β > 0 and r > 0 and r < 1:
                X1 = X[X < θ]
                X2 = X[X >= θ]
                F1 = sp.gammainc(k, θ / β)
                    
                return(len(X1) * (np.log(r) - np.log(F1) - np.log(sp.gamma(k)) - \
                                  k * np.log(β)) - sum(X1) / β +\
                       (k-1) * sum(np.log(X1)) + len(X2) *(np.log(1-r) +\
                        np.log(α) + α * np.log(θ)) - (α + 1) * sum(np.log(X2))
                       )
            else: 
                return(-np.inf)
            
        else:
            return(-np.inf)
    return logp

def logd_gam_par(parms):
    """
    density function of the Gamma-Pareto model.

    Parameters
    ----------
    parms : ndArray 
        particles.

    Returns
    -------
    function
    Allows the evaluation of the density functions for multiple parameter
    values.
    """
    k, α, θ = parms[:,0], parms[:,1], parms[:,2]
    β = θ / (k + α)
    r = α*sp.gamma(k)*  sp.gammainc(k,θ / β) * np.exp(k+α)*(k+α)**(-k) / \
        (1+ α*sp.gamma(k) * sp.gammainc(k, θ / β) * np.exp(k+α)*(k+α)**(-k))
    F1 = sp.gammainc(k, θ / β)
    def logd(x):
        res = np.zeros(len(α))
        s = np.logical_and(β > 0, r > 0, r < 1)
        s1 = np.logical_and(s, x < θ)
        s2 = np.logical_and(s, x >= θ)
        
        res1 = np.log(r[s1]) - np.log(F1[s1]) - np.log(sp.gamma(k[s1])) - \
            k[s1] * np.log(β[s1]) - x / β[s1] + (k[s1]-1) * np.log(x)

        res2 = (np.log(1-r[s2]) + np.log(α[s2]) + α[s2] * \
                np.log(θ[s2])) - (α[s2] + 1) * np.log(x)
        
        res[np.where(s1)] = res1
        res[np.where(s2)] = res2
        res[np.where(np.invert(s))] = -np.inf
        return(res)
    return logd


def sim_wei_par(n, k, α, θ):
    """
    Sample from a Weibull-Pareto model.

    Parameters
    ----------
    n : int 
        sample size.
    k : float
        shape parameter of the Weibull distribution.
    α : float
        Tail index of the Pareto distribution.

    Returns
    -------
    array
    A sample drawn from the Weibull-Pareto distribution.
    
    Example
    -------
    k, α, θ = 1/2, 1/2, 5 
    X = sim_wei_par(1000, k, α, θ)
    """
    β = (k / (k + α))**(1 / k) * θ
    r = (α / θ)*(1 - np.exp(-(k + α) / k))\
    / (α / θ + (k / θ)*np.exp(-(k + α) / k))
    weib_rv = st.weibull_min(k)
    par_rv = st.pareto(α)
    binom_rv =  st.binom(1, r)
    par_rvs = θ * par_rv.rvs(size = n)
    binom_rvs = binom_rv.rvs(size = n)
    weib_rvs = β * weib_rv.ppf(weib_rv.cdf(θ / β) *\
                               np.random.uniform(size = n))
    return(binom_rvs * weib_rvs + (1 - binom_rvs) * par_rvs)


def logp_wei_par(X):
    """
    Likelihood function of the Weibull-Pareto model.

    Parameters
    ----------
    X : Array 
        Insurance losses.

    Returns
    -------
    function
    Allows the evaluation of the likelihood in the parameters provided the 
    data.
    
    Example
    -------
    k, α, θ = 1/2, 1/2, 5 
    X = sim_wei_par(1000, k, α, θ)
    logp = logp_wei_par(X)
    logp(np.array([k, α, θ)])
    costFn = lambda parms: -logp(parms)
    bnds = ((0, None), (0, None), (0, None))
    θ0 = (1, 1, 1)
    minRes = minimize(costFn, θ0,bounds=bnds)
    minRes
    """
    # parms = particles.to_numpy()[4]
    def logp(parms):
        k, α, θ = tuple(parms)
        
        if np.all(parms > 0):
            β = (k / (k + α))**(1 / k) * θ
            r = (α / θ)*(1 - np.exp(-(k + α) / k)) / (α / θ + (k / θ) *\
                                                      np.exp(-(k+α)/k))
            if β > 0 and r > 0 and r < 1:
                X1 = X[X < θ]
                X2 = X[X >= θ]
                F1 = 1 - np.exp(-(θ / β)**k)
                    
                return(len(X1) * \
                       ( np.log(r) + np.log(k) - k * np.log(β) ) + \
                       (k-1) * sum(np.log(X1)) - sum( (X1/ β)**k ) -\
                           len(X1) * np.log(F1) + len(X2) *(np.log(1-r) +\
                        np.log(α) + α * np.log(θ)) - (α + 1) * sum(np.log(X2))
                       )
            else: 
                return(-np.inf)
            
        else:
            return(-np.inf)
    return logp



def logd_wei_par(parms):
    """
    density function of the Weibull-Pareto model.

    Parameters
    ----------
    parms : ndArray 
        particles.

    Returns
    -------
    function
    Allows the evaluation of the density functions for multiple parameter
    values.
    """
    k, α, θ = parms[:,0], parms[:,1], parms[:,2]
    β = (k / (k + α))**(1 / k) * θ
    r = (α / θ)*(1 - np.exp(-(k + α) / k)) / (α / θ + (k / θ) * \
                                                      np.exp(-(k+α)/k))
    F1 = 1 - np.exp(-(θ / β)**k)
    def logd(x):
        res = np.zeros(len(α))
        s = np.logical_and(β > 0, r > 0, r < 1)
        s1 = np.logical_and(s, x < θ)
        s2 = np.logical_and(s, x >= θ)
        
        
        res1 = (np.log(r[s1]) + np.log(k[s1]) - k[s1] * np.log(β[s1])) + \
            (k[s1]-1) * np.log(x) -  (x/ β[s1]) ** k[s1] - \
                np.log(F1[s1])

        res2 = (np.log(1-r[s2]) + np.log(α[s2]) + α[s2] * \
                np.log(θ[s2])) - (α[s2] + 1) * np.log(x)
        
        res[np.where(s1)] = res1
        res[np.where(s2)] = res2
        res[np.where(np.invert(s))] = - np.inf
        return(res)
    return logd

def phi(z):
    """
    Cdf of unit normal distribution

    Parameters
    ----------
    z : Float

    Returns
    -------
    CDF of unit normal distribution
    """
    return( 1 / 2 * (1 + sp.erf(z /np.sqrt(2))))

def sim_lnorm_par(n, σ, α, θ):
    """
    Sample from a lognormal-Pareto model.

    Parameters
    ----------
    n : int 
        sample size.
    σ : float
        shape parameter of the lognormal distribution.
    α : float
        Tail index of the Pareto distribution.
    θ: float
        Threshold parameter

    Returns
    -------
    array
    A sample drawn from the lognormal-Pareto distribution.
    
    Example
    -------
    n, σ, α, θ =10, 1/2, 1/2, 5 
    X = sim_lnorm_par(n, σ, α, θ)
    """
    μ = np.log(θ) - α * σ**2
    
    r = (α * σ  *np.sqrt(2* ma.pi) *phi(α * σ) ) /  \
                (α * σ  *np.sqrt(2* ma.pi) *phi(α * σ) + np.exp(-(α*σ)**2 / 2)) 
    
    lnorm_rv = st.lognorm(s = σ, scale = np.exp(μ))
    
    par_rv = st.pareto(α)
    binom_rv =  st.binom(1, r)
    par_rvs = θ * par_rv.rvs(size = n)
    binom_rvs = binom_rv.rvs(size = n)
    lnorm_rvs = lnorm_rv.ppf(lnorm_rv.cdf(θ) *\
                               np.random.uniform(size = n))
    return(binom_rvs * lnorm_rvs + (1 - binom_rvs) * par_rvs)

def logp_lnorm_par(X):
    """
    Likelihood function of the lognormal-Pareto model.

    Parameters
    ----------
    X : Array 
        Insurance losses.

    Returns
    -------
    function
    Allows the evaluation of the likelihood in the parameters provided the 
    data.
    
    Example
    -------
    n, σ, α, θ =100, 1/2, 1/2, 5
    X = sim_lnorm_par(n, σ, α, θ)
    logp = logp_lnorm_par(X)
    logp(np.array([σ, α, θ]))
    costFn = lambda parms: -logp(parms)
    bnds = ((0, None), (0, None), (0, None))
    θ0 = (1, 1, 3)
    minRes = minimize(costFn, θ0,bounds=bnds)
    minRes
    """
    def logp(parms):
        σ, α, θ = tuple(parms)
        
        if np.all(parms > 0):
            μ = np.log(θ) - α * σ**2
            r = (α * σ  *np.sqrt(2* ma.pi) *phi(α * σ) ) /  \
                (α * σ  *np.sqrt(2* ma.pi) *phi(α * σ) + np.exp(-(α*σ)**2 / 2))
            if r > 0 and r < 1:
                X1 = X[X < θ]
                X2 = X[X >= θ]
                F1 = phi(α * σ)
                    
                return(len(X1) * (np.log(r) - np.log(F1 * σ * np.sqrt(2 * ma.pi)))\
                       - sum(np.log(X1)) - sum((np.log(X1) - μ)**2) / 2 / σ**2 \
                           + len(X2) *(np.log(1-r) + np.log(α) + α * np.log(θ))\
                               - (α + 1) * sum(np.log(X2))
                       )
            else: 
                return(-np.inf)
            
        else:
            return(-np.inf)
    return logp

def logd_lnorm_par(parms):
    """
    density function of the lognormal-Pareto model.

    Parameters
    ----------
    parms : ndArray 
        particles.

    Returns
    -------
    function
    Allows the evaluation of the density functions for multiple parameter
    values.
    """
    σ, α, θ = parms[:,0], parms[:,1], parms[:,2]
    μ = np.log(θ) - α * σ**2
    r = (α * σ  * np.sqrt(2* ma.pi) *phi(α * σ) ) /  \
        (α * σ  * np.sqrt(2* ma.pi) *phi(α * σ) + np.exp(-(α*σ)**2 / 2))
    F1 = phi(α * σ)
    def logd(x):
        
        s = np.logical_and(r > 0, r < 1)
        s1 = np.logical_and(s, x < θ)
        s2 = np.logical_and(s, x >= θ)
        res = np.zeros(len(r))
        
        res1 = (np.log(r[s1]) - np.log(F1[s1] * σ[s1] * np.sqrt(2 * ma.pi)))\
                        - np.log(x) - (np.log(x) - μ[s1])**2 / 2 / σ[s1]**2

        res2 = (np.log(1-r[s2]) + np.log(α[s2]) + α[s2] * np.log(θ[s2])) - (α[s2] + 1) * np.log(x)
        res[np.where(s1)] = res1
        res[np.where(s2)] = res2
        res[np.where(np.invert(s))] = -np.inf
        return(res)
    return logd

def logp_gamma(X):
    """
    Likelihood function of the exponential model.

    Parameters
    ----------
    X : Array 
        Insurance losses.

    Returns
    -------
    function
    Allows the evaluation of the likelihood in the parameters provided the 
    data.
    
    Example
    -------
    α, β = 3, 1 / 3
    gamma_rv = st.gamma(α)
    X =  gamma_rv.rvs(size = 100) /β 
    logp = logp_gamma(X)
    logp(α, β)
    """
    def logp(parms):
        α, β = tuple(parms)
        if np.all(parms > 0) :
            return(len(X) * α * np.log(β) - sum(X) * β + (α-1) * sum(np.log(X))
                   -len(X) * np.log(sp.gamma(α)))
        else:
            return(-np.inf)
    return logp


def logp_wrap(X, loss_model):
    """
    Set the likelihood function for the chosen model.

    Parameters
    ----------
    X : Array 
        Insurance losses.
    loss_model: string
        name of the model

    Returns
    -------
    function
    Allows the evaluation of the likelihood in the parameters provided the 
    data.
    
    Example
    -------
    """
    if loss_model == "wei-par":
        return(logp_wei_par(X))
    elif loss_model == "lnorm-par":
        return(logp_lnorm_par(X))
    elif loss_model == "gam-par":
        return(logp_gam_par(X))
    elif loss_model == "gamma":
        return(logp_gamma(X))

def logd_wrap(parms, loss_model):
    """
    Set the density function for the chosen model.

    Parameters
    ----------
    X : Array 
        Insurance losses.
    loss_model: string
        name of the model

    Returns
    -------
    function
    Allows the evaluation of the likelihood in the parameters provided the 
    data.
    
    Example
    -------
    """
    if loss_model == "wei-par":
        return(logd_wei_par(parms))
    elif loss_model == "lnorm-par":
        return(logd_lnorm_par(parms))
    elif loss_model == "gam-par":
        return(logd_gam_par(parms))

def mle_estimate(X, loss_model, parms_names):
    """
    Provide the mle for the chosen model.

    Parameters
    ----------
    X : Array 
        Insurance losses.
    loss_model: string
        name of the model
    parms_names: array
        name of the parameters

    Returns
    -------
    DatFrame
    Parameter estimate and the likelihood function
    
    Example
    -------
    k, α, θ = 1/2, 1/2, 5 
    X, loss_model = sim_wei_par(100, k, α, θ), "wei-par"
    mle_estimate(X, loss_model)

    """
    logp = logp_wrap(X, loss_model)
    
    res = pd.DataFrame({parms_names[0]:[],parms_names[1]:[],parms_names[2]:[], 'log_lik':[]})
    for j in range(len(X)):
        θ = np.sort(X)[j]
        def logp_fixed_θ(parms):
            k, α = parms
            return(logp(np.array([k, α, θ])))
        costFn = lambda parms: -logp_fixed_θ(parms)
        bnds = ((0, None), (0, None))
        θ0 = (1, 1)
        try:
            minRes = minimize(costFn, θ0,bounds=bnds)
        except:
            minRes.x = np.array([1,1])
        res = pd.concat([res,
                   pd.DataFrame({parms_names[0]:[minRes.x[0]],parms_names[1]:[minRes.x[1]],'θ':[θ],
                                 'log_lik':logp(np.append(minRes.x,θ))})])
    
    return(res[res['log_lik'] == res['log_lik'].max()])




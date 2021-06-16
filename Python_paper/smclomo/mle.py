# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 10:01:49 2021

@author: pierr
"""
import numpy as np
from scipy.special import gamma
import math as ma
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import pandas as pd
from .loss_distribution import logp_wrap

def infer_gamma(X, θ0):    
    n = len(X)
    ΣX = sum(X)
    ΣlogX = sum(np.log(X))
    def logp(θ):
        k, δ = θ
        return - n * np.log(gamma(k)) - n * k * np.log(δ) + (k - 1) * ΣlogX - ΣX / δ 
     
    costFn = lambda θ: -logp(θ)
    bnds = ((0, None), (0, None))
    
    minRes = minimize(costFn, θ0, bounds=bnds)
    
    BIC = len(θ0) * np.log(n) - 2 * logp(minRes.x) 
    AIC = 2 * len(θ0) - 2 * logp(minRes.x) 
    return minRes.x[0], minRes.x[1], BIC, AIC

def infer_lnorm(X):
    n = len(X)
    θHat =  np.std(np.log(X)), np.mean(np.log(X))
    def logp(θ):
        σ, mu = θ
        return - n * np.log(σ * (2 * ma.pi)**(1/2)) - sum(np.log(X)) - sum((np.log(X) - mu)**2 / 2 / σ**2)
    BIC = len(θHat) * np.log(n) - 2 * logp(θHat)
    AIC= 2 * len(θHat) - 2 * logp(θHat)
    
    return θHat[0], θHat[1], BIC, AIC

def infer_weib(X, θ0):
    n = len(X) 
    def logp(θ):
        k, δ = θ
        return n * np.log(k) - n * k * np.log(δ) + (k-1) * sum(np.log(X)) - sum((X / δ)**k)
    
    costFn = lambda θ: -logp(θ)
    bnds = ((0, None), (0, None))
    
    minRes = minimize(costFn, θ0, bounds=bnds)
    
    BIC = len(θ0) * np.log(n) - 2 * logp(minRes.x)
    AIC = 2 * len(θ0) - 2 * logp(minRes.x)
    return minRes.x[0], minRes.x[1], BIC, AIC

def infer_par(X):
    # sevs is a vector of severities
    # θ0 is a first guess of the parameter values    
    n = len(X) 
    def logp(θ):
        γ, α = θ
        return n * np.log(α) + n * α * np.log(γ) - (α + 1) * sum(np.log(X))
    
    γ, α = min(X), 1/np.mean(np.log(X/min(X)))
    
    BIC = 2 * np.log(n) - 2 * logp((γ, α)) 
    AIC = 2*2 - 2 * logp((γ, α))
    return γ, α, BIC, AIC



def mle_composite(X, θ0, loss_model):
    n = len(X) 
    logp = logp_wrap(X, loss_model)
    costFn = lambda parms: -logp(parms)
    bnds = ((0, None), (0, None), (0, None))
    minRes = minimize(costFn, θ0,bounds=bnds)
    BIC = len(θ0) * np.log(n) - 2 * logp(minRes.x)
    AIC = 2 * len(θ0) - 2 * logp(minRes.x)
    return(minRes.x, BIC, AIC)

def qq_plot(X, X0, color):
    qs = np.arange(0.01,0.99, 0.01)
    
    df_quantile = pd.DataFrame({
        'emp_quant' : pd.Series(X).quantile(qs),
        'lnorm_quant' : pd.Series(X0).quantile(qs)
    })
    df_quantile
    # Quantile-quantile plots
    fig, axs = plt.subplots(1, 1, figsize=(2.5, 2.5))
    
    plt.plot(df_quantile.emp_quant, df_quantile.lnorm_quant, c= color, lw=3)
    
    x = np.linspace(0, max(df_quantile.emp_quant), 100)
    
    plt.plot(x, x, ":k")
    plt.xticks([])
    plt.yticks([])
    plt.xlabel("Quantiles empiriques")
    plt.ylabel("Quantiles théoriques")
    
def qq_plot_en(X, X0, color):
    qs = np.arange(0.01,0.99, 0.01)
    
    df_quantile = pd.DataFrame({
        'emp_quant' : pd.Series(X).quantile(qs),
        'lnorm_quant' : pd.Series(X0).quantile(qs)
    })
    df_quantile
    # Quantile-quantile plots
    fig, axs = plt.subplots(1, 1, figsize=(2.5, 2.5))
    
    plt.plot(df_quantile.emp_quant, df_quantile.lnorm_quant, c= color, lw=3)
    
    x = np.linspace(0, max(df_quantile.emp_quant), 100)
    
    plt.plot(x, x, ":k")
    plt.xticks([])
    plt.yticks([])
    plt.xlabel("Empirical")
    plt.ylabel("Theoretical")






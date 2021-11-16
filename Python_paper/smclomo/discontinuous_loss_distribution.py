# -*- coding: utf-8 -*-
"""
Created on Sun Nov 14 18:32:17 2021

@author: pierr
"""

import scipy.special as sp
import math as ma
import numpy as np
import scipy.stats as st

import numba as nb
from .loss_distribution import reg_inc_gamma

def sim_dis_gam_par(n, r, m, α, γ, p):
    """
    Sample from a discontinuous Gamma-Pareto model.

    Parameters
    ----------
    n : int 
        sample size.
    r : float
        shape parameter of the Gamma distribution.
    m : float
        scale parameter of the Gamma distribution.
    α : float
        Tail index of the Pareto distribution.
    γ : float
        Threshold parameter of the composite model.
    p: float
        mixing parameter of the composite model

    Returns
    -------
    array
    A sample drawn from the dicontinuous gamma-Pareto distribution.
    
    Example
    -------
    n, r, m, α, γ, p = 10, 2, 3, 1/2, 5, 1/3
    X = sim_dis_gam_par(n, r, m, α, γ, p)
    """    
    gamma_rv = st.gamma(r)
    par_rv = st.pareto(α)
    binom_rv =  st.binom(1, p)
    par_rvs = γ * par_rv.rvs(size = n)
    binom_rvs = binom_rv.rvs(size = n)
    gamma_rvs = m * gamma_rv.ppf(sp.gammainc(r, γ / m) *\
                               np.random.uniform(size = n))
    return(binom_rvs * gamma_rvs + (1 - binom_rvs) * par_rvs)

n, r, m, α, γ, p = 10, 2, 3, 1/2, 5, 1/3
X = sim_dis_gam_par(n, r, m, α, γ, p)

def logp_dis_gam_par(X):
    """
    Likelihood function of the discontinuous Gamma-Pareto model.

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
    n, r, m, α, γ, p = 10, 2, 3, 1/2, 5, 1/3
    X = sim_dis_gam_par(n, r, m, α, γ, p)
    logp = logp_dis_gam_par(X)
    logp(np.array([2, 3, 1/2, 5, 1/3]))
    costFn = lambda parms: -logp(parms)
    bnds = ((0, None), (0, None), (0, None))
    θ0 = (1, 1, 1)
    minRes = minimize(costFn, θ0,bounds=bnds)
    minRes
    """
    def logp(parms):
        r, m, α, γ, p = parms
        
        if np.all(parms > 0):
            if p > 0 and p < 1:
                X1 = X[X < γ]
                X2 = X[X >= γ]
                F1 = reg_inc_gamma(r, γ / m)
                    
                return(len(X1) * (np.log(p) - np.log(F1) - np.log(ma.gamma(r)) - \
                                  r * np.log(m)) - np.sum(X1) / m +\
                       (r-1) * np.sum(np.log(X1)) + len(X2) *(np.log(1-p) +\
                        np.log(α) + α * np.log(γ)) - (α + 1) * np.sum(np.log(X2))
                       )
            else: 
                return(-np.inf)
            
        else:
            return(-np.inf)
    return nb.jit(nopython = True)(logp)


# n, r, m, α, γ, p = 100, 2, 3, 1/2, 5, 1/3
# X = sim_dis_gam_par(n, r, m, α, γ, p)
# logp = logp_dis_gam_par(X)
# logp(np.array([2, 3, 1/2, 5, 1/3]))
# costFn = lambda parms: -logp(parms)
# bnds = ((0, None), (0, None), (0, None), (0,None), (0,1))
# θ0 = (3, 1, 1/5,3,1/5)
# minRes = minimize(costFn, θ0,bounds=bnds)
# minRes

# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 09:23:14 2021

@author: pierr
"""
import pandas as pd
import numpy as np
from .prior_distribution import logp_prior_wrap
from .loss_distribution import logp_wrap
from .move import Gibbs_move
from .smc import smc_likelihood_annealing, smc_data_by_batch

def fit_model_gibbs(X, loss_model, model_prior, a, b, popSize, step_size,  
                    init_parms, parms_names):
    log_prob_prior, log_prob = logp_prior_wrap(model_prior, a, b),\
        logp_wrap(X, loss_model)
    trace, acceptance = Gibbs_move(popSize * 2, step_size, log_prob,\
                                   log_prob_prior, init_parms, 1, len(parms_names))
    trace_gibbs = pd.DataFrame(trace).iloc[int(popSize):int(popSize * 2)]
    trace_gibbs.columns = parms_names
    return(trace_gibbs, acceptance)    


def fit_composite_models_smc(X, loss_models, model_prior, a, b, popSize, verbose, smc_method, paralell, nproc):
    traces = {}
    res_df = pd.DataFrame(columns = ["loss_model","log_marg", "DIC", "WAIC", 
                          "shape", "tail", "thres"])
    for loss_model in loss_models:
        if verbose:
            print("Fitting "+loss_model+ " model")
        if smc_method == "likelihood_anealing":
            trace_smc, log_marg, DIC, WAIC = smc_likelihood_annealing(X, loss_model, 
                                                                  ["shape", "tail", "thres"], 
                                                                  popSize, 
                                                                  model_prior, a, b, 1/2,
                                                                  0.99, 25, 1e-6, 
                                                                  paralell, nproc, 
                                                                  False)
        else:
            trace_smc, log_marg, DIC, WAIC = smc_data_by_batch(X, loss_model, 
                                                                  ["shape", "tail", "thres"], 
                                                                  popSize, 
                                                                  model_prior, a, b, 1/2,
                                                                  0.99, 25, 
                                                                  paralell, nproc, 
                                                                  False)
        traces[loss_model] = trace_smc
        df1 = pd.DataFrame(trace_smc.mean()).T
        df2 = pd.DataFrame(np.array([log_marg, DIC, WAIC]).reshape(1,3), 
                     columns = ["log_marg", "DIC", "WAIC"], 
                     index = [loss_model]).rename_axis("loss_model").reset_index()
        df2 = df2.join(df1)
        res_df = pd.concat([res_df, df2])
    res_df["model_evidence"] = (np.exp(res_df["log_marg"].values - res_df["log_marg"].max())) / \
     sum(np.exp(res_df["log_marg"].values - res_df["log_marg"].max()))
    return(traces, res_df)

def compare_models(X, model_prior, a, b, popSize, verbose, smc_method):
    res_df = pd.DataFrame(columns = ["loss_model","log_marg", "DIC", "WAIC", 
                          "shape", "tail", "thres"])

    for loss_model in ["lnorm-par", "wei-par", "gam-par"]:
        if verbose:
            print("Analysing "+loss_model+ " model")
        if smc_method == "likelihood_anealing":
            trace_smc, log_marg, DIC, WAIC = smc_likelihood_annealing(X, loss_model, 
                                                                  ["shape", "tail", "thres"], 
                                                                  popSize, 
                                                                  model_prior, a, b, 1/2,
                                                                  0.99, 25, 1e-6, 
                                                                  False, 4, 
                                                                  False)
        else:
            trace_smc, log_marg, DIC, WAIC = smc_data_by_batch(X, loss_model, 
                                                                  ["shape", "tail", "thres"], 
                                                                  popSize, 
                                                                  model_prior, a, b, 1/2,
                                                                  0.99, 25, 
                                                                  False, 4, 
                                                                  False)
        df1 = pd.DataFrame(trace_smc.mean()).T
        df2 = pd.DataFrame(np.array([log_marg, DIC, WAIC]).reshape(1,3), 
                     columns = ["log_marg", "DIC", "WAIC"], 
                     index = [loss_model]).rename_axis("loss_model").reset_index()
        df2 = df2.join(df1)
       
        res_df = pd.concat([res_df, df2])
        
    res_df["model_evidence"] = (np.exp(res_df["log_marg"].values - res_df["log_marg"].max())) / \
     sum(np.exp(res_df["log_marg"].values - res_df["log_marg"].max()))
          
    return(res_df)

    


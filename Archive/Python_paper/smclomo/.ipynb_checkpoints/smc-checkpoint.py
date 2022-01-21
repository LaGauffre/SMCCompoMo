# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 17:21:21 2020

@author: pierr
"""

import pandas as pd
import numpy as np
import scipy.special as sp
import scipy.stats as st
from joblib import Parallel, delayed
from .loss_distribution import logp_wrap, logd_wrap
from .prior_distribution import sim_prior_wrap, logp_prior_wrap
from .temperature import temperature_search, batch_size_search
from .move import Gibbs_move

def smc_likelihood_annealing(X, loss_model, parms_names, popSize, model_prior, a, b, ρ,
                             c,n_step_max, err, paralell, n_proc, verbose):
    """
    Sequential Monte Carlo Sampler of the posterior distribution.

    Parameters
    ----------
    X: array
        loss data required to evaluate the likelihood
    loss_model: string
        loss model being fitted
    parms_names: array
        names of the loss model parameters, 
        first the shape parameter of the belly distribution
        second the tail index
        third the threshold between small and large claims
    popSize: int
        number of particles sampled, size of the cloud
    model_prior: string
        prior distribution
    a, b: arrays 
        prior distribution parameters
    ρ: float
        tuning parameter for the target effective sample size 
    move_type: string 
        type of moves to choose in ("Metropolis", "Gibbs", "Indpendent)"
    c : float
        Calibrate the number of steps required so that the probability that 
        each particle is moved at least once equals c.
    n_step_max: int
        limits the number of steps
    err: float
        Temperature threshold
    verbose: boolean
        Whether to print the steps
    
    Returns
    -------
    list
    A list that provides the posterior sample along with the smc estimator of 
    the marginal likelihood.
    
    Example
    -------
    k, α, θ = 1/2, 1/2, 5 
    X = sim_wei_par(10, k, α, θ)
    loss_model, parms_names, popSize, model_prior, a, b, method, ρ, move_type, c, \
    n_step_max = "wei-par", ['k','α', 'θ'], 20, "uniform", np.array([0,0,0]),\
    np.array([10,10,10]), "likelihood annealing", 1/2, "Gibbs", 0.99, 20
    trace, marg_log_lik = smc(X, loss_model, parms_names,
                          100, 
                          model_prior, a, b, 
                          method, ρ, move_type, c,n_step_max)
    trace.mean(), marg_log_lik
    """
    log_prob_prior, log_prob, d = logp_prior_wrap(model_prior, a, b), \
        logp_wrap(X, loss_model), len(parms_names)
    # Generation counter
    g = 0
    if verbose:
        print('Sample generation ' + str(g) + " from the " + str(model_prior) +
          " prior distribution")
    # Initialisation of the particle cloud
    init_cloud = sim_prior_wrap(model_prior, a, b, parms_names, popSize)
    init_cloud['logw'] = np.log(np.ones(popSize))
    init_cloud['W'] = np.ones(popSize) / popSize
    # This initial particle cloud is placed inside a list
    clouds = []
    clouds.append(init_cloud)
    
    # Temperature sequence either true temperature or proportion of observations
    γ_seq = np.array([0])

   
    # We keep on iterating until the temperature reaches 1
    while γ_seq[-1] < 1:
        
        g = g + 1
        particles = clouds[g-1][parms_names].values
        # Updating temperature sequence 
        γ, logw, W, ESS = temperature_search(particles,
                                             log_prob,ρ * popSize, γ_seq[-1],
                                             err)
        
       
        γ_seq = np.append(γ_seq, γ)
       
        cloud_cov = np.cov(particles, 
                           bias = False, 
                           aweights = W, 
                           rowvar = False) * 2.38 / np.sqrt(d)
        
        particles_resampled = particles[np.random.choice(popSize,popSize, p = W)]
        
        def move_particle_trial(particle):
            trace, acceptance = Gibbs_move(1,np.diag(cloud_cov), log_prob, 
                                           log_prob_prior, particle, γ, d)
            return(np.append(trace[-1], np.mean(np.any(acceptance[1:]))))
        
        if paralell:
            res_trial = np.array(Parallel(n_jobs=n_proc)(delayed(move_particle_trial)(i) 
                                           for i in particles_resampled))
        else:    
            res_trial = np.array([move_particle_trial(particle) 
                                  for particle in particles_resampled])
        particles_trial, acc_trial = res_trial[:,0:d], res_trial[:,-1]
        n_steps = int(min(n_step_max,max(2,np.ceil(np.log(1-c) / np.log(1-(np.mean(acc_trial)-1e-6))))))
        def move_particle(particle):
            trace, acceptance = Gibbs_move(n_steps,np.diag(cloud_cov), log_prob, 
                                           log_prob_prior, particle, γ, d)
            return(np.append(trace[-1], np.mean(np.any(acceptance[1:]))))
        if paralell:
            res = np.array(Parallel(n_jobs=n_proc)(delayed(move_particle)(i) 
                                           for i in particles_trial))
        else:    
            res = np.array([move_particle(particle) for particle in particles_trial])
        particles_moved, acc_rate = res[:,0:d], res[:,-1]
        if verbose:
            print('Generation: ' + str(g) + " ;temperature: "+str(γ_seq[-1])+
                  " ;ESS: "+str(ESS)+
               " ;steps:" + str(n_steps+1) + " ;particle moved: "+
               str(np.mean(acc_rate) * 100) + "%" )
      
        
        cloud = pd.DataFrame(particles_moved)
        cloud.columns = parms_names
        # Updating unormalized weights
        cloud['logw'] = logw
        # Updating normalized weights
        cloud['W'] = W
        clouds.append(cloud)

    marginal_log_likelihood = sum([ sp.logsumexp(cloud['logw'] - np.log(popSize)) 
                                            for cloud in clouds[1:g+1]])
    
    log_probs = [log_prob(particle) for particle in particles_moved]
    DIC =  - 2* log_prob(np.mean(particles_moved, axis = 0)) + \
    2* (2* np.mean(log_probs) - 2* log_prob(np.mean(particles_moved, axis = 0)))
    logd = logd_wrap(particles_moved, loss_model)
    logds = np.array([logd(x) for x in X])
    WAIC = - 2*( 
        sum(np.log(np.mean(np.exp(logds), axis = 1))) -
                sum(np.var(logds, axis = 1))
                )  
    return(clouds[-1][parms_names], marginal_log_likelihood, DIC, WAIC)


def smc_data_by_batch(X, loss_model, parms_names, popSize, model_prior, a, b, ρ, 
                      c, n_step_max, paralell, n_proc, verbose):
    """
    Sequential Monte Carlo Sampler of the posterior distribution.

    Parameters
    ----------
    X: array
        loss data required to evaluate the likelihood
    loss_model: string
        loss model being fitted
    parms_names: array
        names of the loss model parameters, 
        first the shape parameter of the belly distribution
        second the tail index
        third the threshold between small and large claims
    popSize: int
        number of particles sampled, size of the cloud
    model_prior: string
        prior distribution
    a, b: arrays 
        prior distribution parameters
    ρ: float
        tuning parameter for the target effective sample size 
    move_type: string 
        type of moves to choose in ("Metropolis", "Gibbs", "Indpendent)"
    c : float
        Calibrate the number of steps required so that the probability that 
        each particle is moved at least once equals c.
    n_step_max: int
        limits the number of steps
    verbose: boolean
        Whether to print the steps
    
    Returns
    -------
    list
    A list that provides the posterior sample along with the smc estimator of 
    the marginal likelihood.
    
    Example
    -------
    k, α, θ = 1/2, 1/2, 5 
    X = sim_wei_par(10, k, α, θ)
    loss_model, parms_names, popSize, model_prior, a, b, method, ρ, move_type, c, \
    n_step_max = "wei-par", ['k','α', 'θ'], 20, "uniform", np.array([0,0,0]),\
    np.array([10,10,10]), "likelihood annealing", 1/2, "Gibbs", 0.99, 20
    trace, marg_log_lik = smc(X, loss_model, parms_names,
                          100, 
                          model_prior, a, b, 
                          method, ρ, move_type, c,n_step_max)
    trace.mean(), marg_log_lik
    """
    log_prob_prior, d = logp_prior_wrap(model_prior, a, b), len(parms_names)
    # Generation counter
    g = 0
    if verbose:
        print('Sample generation ' + str(g) + " from the " + str(model_prior) +
          " prior distribution")
    # Initialisation of the particle cloud
    init_cloud = sim_prior_wrap(model_prior, a, b, parms_names, popSize)
    init_cloud['logw'] = np.log(np.ones(popSize))
    init_cloud['W'] = np.ones(popSize) / popSize
    # This initial particle cloud is placed inside a list
    clouds = []
    clouds.append(init_cloud)
    
    # sequence of data batch size
    n_seq = np.array([0])

   
    # We keep on iterating until the temperature reaches 1
    while n_seq[-1] < len(X):
        
        g = g + 1
       
        # Updating temperature sequence 
        particles = clouds[g-1][parms_names].values
        
        n, logw, W, ESS = batch_size_search(particles, ρ * popSize, n_seq[-1], X, loss_model)
        
        
       
        n_seq = np.append(n_seq, n)
       
        cloud_cov = np.cov(particles, 
                           bias = False, 
                           aweights = W, 
                           rowvar = False) * 2.38 / np.sqrt(d)
        
        
        particles_resampled = particles[np.random.choice(popSize,popSize, p = W)]
        log_prob = logp_wrap(X[0:n], loss_model)
        def move_particle_trial(particle):
            trace, acceptance = Gibbs_move(1,np.diag(cloud_cov), log_prob, 
                                           log_prob_prior, particle, 1, d)
            return(np.append(trace[-1], np.mean(np.any(acceptance[1:]))))
        
        if paralell:
            res_trial = np.array(Parallel(n_jobs=n_proc)(delayed(move_particle_trial)(i) 
                                           for i in particles_resampled))
        else:    
            res_trial = np.array([move_particle_trial(particle) 
                                  for particle in particles_resampled])
        particles_trial, acc_trial = res_trial[:,0:d], res_trial[:,-1]
        n_steps = int(min(n_step_max,max(2,np.ceil(np.log(1-c) / np.log(1-(np.mean(acc_trial)-1e-6))))))
        def move_particle(particle):
            trace, acceptance = Gibbs_move(n_steps,np.diag(cloud_cov), log_prob, 
                                           log_prob_prior, particle, 1, d)
            return(np.append(trace[-1], np.mean(np.any(acceptance[1:]))))
        if paralell:
            res = np.array(Parallel(n_jobs=n_proc)(delayed(move_particle)(i) 
                                           for i in particles_trial))
        else:    
            res = np.array([move_particle(particle) for particle in particles_trial])
        particles_moved, acc_rate = res[:,0:d], res[:,-1]
        if verbose:
            print('Generation: ' + str(g) + " ;batch size: "+str(n_seq[-1])+
                  " ;ESS: "+str(ESS)+
               " ;steps:" + str(n_steps+1) + " ;particle moved: "+
               str(np.mean(acc_rate) * 100) + "%")
      
        # print(np.mean(acc_rate2==0),np.mean(acc_rate2), n_steps)
        cloud = pd.DataFrame(particles_moved)
        cloud.columns = parms_names
        # Updating unormalized weights
        cloud['logw'] = logw
        # Updating normalized weights
        cloud['W'] = W
        clouds.append(cloud)
    marginal_log_likelihood = sum([ sp.logsumexp(cloud['logw'] - np.log(popSize)) 
                                            for cloud in clouds[1:g+1]])
    # marginal_log_likelihood = sum(np.log(([np.exp(cloud['w'].values).mean() 
    #                                         for cloud in clouds[1:g+1]])))
    log_probs = [log_prob(particle) for particle in particles_moved]
    DIC =  - 2* log_prob(np.mean(particles_moved, axis = 0)) + \
    2* (2* np.mean(log_probs) - 2* log_prob(np.mean(particles_moved, axis = 0)))
    logd = logd_wrap(particles_moved, loss_model)
    logds= np.array([logd(x) for x in X])
    WAIC = - 2*(sum(np.log(np.mean(np.exp(logds), axis = 1))) - 
                sum(np.var(logds, axis = 1)))   
    
    
    return(clouds[-1][parms_names], marginal_log_likelihood, DIC, WAIC)


def log_marg_bridge_sampling(logp, logp_prior, trace, r_init, err):
    
    th_1 = trace.values
    MN_rv = st.multivariate_normal(mean=trace.mean(), cov=trace.cov())
    th_2 = MN_rv.rvs(len(th_1))
    
    log_probs_prior_th_1 = np.array([logp_prior(particle) for particle in th_1])
    logps_th_1 = np.array([logp(particle) for particle in th_1])
    log_probs_prior_th_2 = np.array([logp_prior(particle) for particle in th_2])
    logps_th_2 = np.array([logp(particle) for particle in th_2])
    
    eta_1_th_1 = np.exp(logps_th_1+log_probs_prior_th_1)
    eta_1_th_2 = np.exp(logps_th_2+log_probs_prior_th_2)
    eta_2_th_1 = MN_rv.pdf(th_1)
    eta_2_th_2 = MN_rv.pdf(th_2)
    r = [r_init]
    while True:
        r_new = np.sum(eta_1_th_2 / (eta_1_th_2 +  r[-1]*eta_2_th_2)) / np.sum(eta_2_th_1 / (eta_1_th_1 +  r[-1]*eta_2_th_1))
        if abs(np.log(r[-1]) - np.log(r_new)) < err : 
            r.append(r_new)
            break
        else:
            r.append(r_new)
    return(np.log(r[-1]))


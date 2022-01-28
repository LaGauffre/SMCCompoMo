# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 10:33:59 2021

@author: pierr
"""
import bayes_splicing as bs

import pandas as pd
import numpy as np
import scipy.stats as st
import scipy.special as sp
import math as ma
from scipy.optimize import minimize
from joblib import Parallel, delayed
import scipy as sc
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from time import time
from IPython.display import display_html
import pickle

import datetime as dat

# Importation of the model comparison data
well_spec_250 = pd.read_csv("../../Data/Simulations/simu_well_spec_250.csv")
well_spec_500 = pd.read_csv("../../Data/Simulations/simu_well_spec_500.csv")
well_spec_1000 = pd.read_csv("../../Data/Simulations/simu_well_spec_1000.csv")
well_spec_2000 = pd.read_csv("../../Data/Simulations/simu_well_spec_2000.csv")
well_spec_df = pd.concat([well_spec_250, well_spec_500,well_spec_1000, well_spec_2000])

# Importation of the simulated data itsef
Xs_250 = pickle.load( open( "../../Data/Simulations/sim_data_250.obj", "rb" ) )
Xs_500 = pickle.load( open( "../../Data/Simulations/sim_data_500.obj", "rb" ) )
Xs_1000 = pickle.load( open( "../../Data/Simulations/sim_data_1000.obj", "rb" ) )
Xs_2000 = pickle.load( open( "../../Data/Simulations/sim_data_2000.obj", "rb" ) )

# Model for the bulk distribution
body_model_names = ["Exp", "Gamma", "Weibull", "Lognormal", "Inverse-Weibull", "Inverse-Gamma", "Inverse-Gaussian", "Lomax", "Log-Logistic", "Burr"]
body_model_param_names = [ ["λ1"], ["r1", "m1"], ["k1", "β1"],
                          ["μ1", "σ1"], ["k1", "β1"], ["r1", "m1"], ["μ1", "λ1"], ["α1", "σ1"], ["β1", "σ1"], ["α1", "β1", "σ1"] ]

# Prior distributions over the parameters of the bulk distribution
body_model_priors= [ 
    [bs.prior_model('gamma',body_model_param_names[0][0], 1, 1)], 
     [bs.prior_model('gamma',body_model_param_names[1][0], 1, 1), bs.prior_model('gamma',body_model_param_names[1][1], 1, 1)],
    [bs.prior_model('gamma',body_model_param_names[2][0], 1, 1), bs.prior_model('gamma',body_model_param_names[2][1], 1, 1)],
    [bs.prior_model('normal',body_model_param_names[3][0], 0, 0.5), bs.prior_model('gamma',body_model_param_names[3][1], 1, 1)],
     [bs.prior_model('gamma',body_model_param_names[4][0], 1, 1), bs.prior_model('gamma',body_model_param_names[4][1], 1, 1)], 
    [bs.prior_model('gamma',body_model_param_names[5][0], 1, 1), bs.prior_model('gamma',body_model_param_names[5][1], 1, 1)], 
    [bs.prior_model('gamma',body_model_param_names[6][0], 1, 1), bs.prior_model('gamma',body_model_param_names[6][1], 1, 1)], 
    [bs.prior_model('gamma',body_model_param_names[7][0], 1, 1), bs.prior_model('gamma',body_model_param_names[7][1], 1, 1)], 
    [bs.prior_model('gamma',body_model_param_names[8][0], 1, 1), bs.prior_model('gamma',body_model_param_names[8][1], 1, 1)],
    [bs.prior_model('gamma',body_model_param_names[9][0], 1, 1), bs.prior_model('gamma',body_model_param_names[9][1], 1, 1), 
     bs.prior_model('gamma',body_model_param_names[9][2], 1, 1)]
]

# Model for the tail of the distribution
tail_model_names = ["Weibull", "Lognormal", "Log-Logistic", "Lomax", "Burr", "Pareto-Tail", "GPD-Tail", "Inverse-Gamma", "Inverse-Weibull", "Exp", "Gamma"]

tail_model_param_names = [["k2", "β2"], ["μ2", "σ2"], ["β2", "σ2"], ["α2", "σ2"], ["α2", "β2", "σ2"], ["α2"], ["ξ2","σ2"], ["r2", "m2"], ["k2", "β2"], ["λ2"], ["r2", "m2"]]

# Prior distributions over the parameters of the bulk distribution
tail_model_priors= [
                [bs.prior_model('gamma',tail_model_param_names[0][0], 1, 1), bs.prior_model('gamma',tail_model_param_names[0][1], 1, 1)],
                [bs.prior_model('normal',tail_model_param_names[1][0], 0, 0.5), bs.prior_model('gamma',tail_model_param_names[1][1], 1, 1)],
                [bs.prior_model('gamma',tail_model_param_names[2][0], 1, 1), bs.prior_model('gamma',tail_model_param_names[2][1], 1, 1)],
                [bs.prior_model('gamma',tail_model_param_names[3][0], 1, 1), bs.prior_model('gamma',tail_model_param_names[3][1], 1, 1)],
                [bs.prior_model('gamma',tail_model_param_names[4][0], 1, 1), bs.prior_model('gamma',tail_model_param_names[4][1], 1, 1), bs.prior_model('gamma',tail_model_param_names[4][2], 1, 1)],
                [bs.prior_model('gamma',tail_model_param_names[5][0], 1, 1)],
                [bs.prior_model('gamma',tail_model_param_names[6][0], 1, 1), bs.prior_model('gamma',tail_model_param_names[6][1], 1, 1)],
                [bs.prior_model('gamma',tail_model_param_names[7][0], 1, 1), bs.prior_model('gamma',tail_model_param_names[7][1], 1, 1)],
                [bs.prior_model('gamma',tail_model_param_names[8][0], 1, 1), bs.prior_model('gamma',tail_model_param_names[8][1], 1, 1)],
                [bs.prior_model('gamma',tail_model_param_names[9][0], 1, 1)],
                [bs.prior_model('gamma',tail_model_param_names[10][0], 1, 1), bs.prior_model('gamma',tail_model_param_names[10][1], 1, 1)]
]

γ_prior = bs.prior_model('gamma',"γ", 1, 1)

#Splicing model type
splicing_types = ["continuous"]

# Setting the models
fs, f_names, prior_spliced_model = [], [], []
for i in range(len(body_model_names)):
    for j in range(len(tail_model_names)):
        for splicing_type in splicing_types:
            f1, f2 =  bs.loss_model(body_model_names[i], body_model_param_names[i]), bs.loss_model(tail_model_names[j], tail_model_param_names[j])
            fs.append(bs.spliced_loss_model(f1 , f2, splicing_type))
            f_names.append(body_model_names[i] +"_"+ tail_model_names[j]+"_"+splicing_type)
            if splicing_type == "disjoint": 
                prior_spliced_model.append(bs.independent_priors(body_model_priors[i] + tail_model_priors[j] + [γ_prior, p_prior]))
            else:
                prior_spliced_model.append(bs.independent_priors(body_model_priors[i] + tail_model_priors[j] + [γ_prior]))  
for f in fs:
    f.set_ppf(), f.set_cdf(), f.set_pdf() 
f_spliced_dic = dict(zip(f_names, fs))
prior_dic = dict(zip(f_names, prior_spliced_model))

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

import datetime as dat

import pickle
import random
random.seed(1234)

# Model for the bulk distribution
body_model_names = ["Weibull", "Inverse-Weibull"]
body_model_param_names = [ ["k1", "β1"], ["k1", "β1"] ]

# Prior distributions over the parameters of the bulk distribution
body_model_priors= [  
     [bs.prior_model('gamma',body_model_param_names[0][0], 1, 1), bs.prior_model('gamma',body_model_param_names[0][1], 1, 1)],
    [bs.prior_model('gamma',body_model_param_names[1][0], 1, 1), bs.prior_model('gamma',body_model_param_names[1][1], 1, 1)]
]

# Model for the tail of the distribution
tail_model_names = ["Log-Logistic", "Lomax", "Inverse-Weibull"]

tail_model_param_names = [ ["β2", "σ2"], ["α2", "σ2"], ["k2", "β2"]]

# Prior distributions over the parameters of the bulk distribution
tail_model_priors= [
                [bs.prior_model('gamma',tail_model_param_names[0][0], 1, 1), bs.prior_model('gamma',tail_model_param_names[0][1], 1, 1)],
          
                [bs.prior_model('gamma',tail_model_param_names[1][0], 1, 1), bs.prior_model('gamma',tail_model_param_names[1][1], 1, 1)],
                [bs.prior_model('gamma',tail_model_param_names[2][0], 1, 1), bs.prior_model('gamma',tail_model_param_names[2][1], 1, 1)]
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
            f_names.append(body_model_names[i] +"_"+ tail_model_names[j])
            prior_spliced_model.append(bs.independent_priors(body_model_priors[i] + tail_model_priors[j] + [γ_prior]))  
for f in fs:
    f.set_ppf(), f.set_cdf(), f.set_pdf() 
f_spliced_dic = dict(zip(f_names, fs))
prior_dic = dict(zip(f_names, prior_spliced_model))



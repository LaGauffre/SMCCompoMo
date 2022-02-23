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

well_spec_500 = pd.read_csv("../../Data/Simulations/simu_wellspec_500.csv")
well_spec_1000 = pd.read_csv("../../Data/Simulations/simu_wellspec_1000.csv")
well_spec_2000 = pd.read_csv("../../Data/Simulations/simu_wellspec_2000.csv")
well_spec_5000 = pd.read_csv("../../Data/Simulations/simu_wellspec_5000.csv")

well_spec_df = pd.concat([ well_spec_500,well_spec_1000, well_spec_2000, well_spec_5000])

miss_spec_500 = pd.read_csv("../../Data/Simulations/simu_missspec_500.csv")
miss_spec_1000 = pd.read_csv("../../Data/Simulations/simu_missspec_1000.csv")
miss_spec_2000 = pd.read_csv("../../Data/Simulations/simu_missspec_2000.csv")
miss_spec_5000 = pd.read_csv("../../Data/Simulations/simu_missspec_5000.csv")
miss_spec_df = pd.concat([ miss_spec_500,miss_spec_1000, miss_spec_2000,  miss_spec_5000])

# # Importation of the simulated data itsef
# Xs_250 = pickle.load( open( "../../Data/Simulations/sim_data_250.obj", "rb" ) )
# Xs_500 = pickle.load( open( "../../Data/Simulations/sim_data_500.obj", "rb" ) )
# Xs_1000 = pickle.load( open( "../../Data/Simulations/sim_data_1000.obj", "rb" ) )
# Xs_2000 = pickle.load( open( "../../Data/Simulations/sim_data_2000.obj", "rb" ) )

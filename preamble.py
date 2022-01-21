# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 10:33:59 2021

@author: pierr
"""
import bayessplicedmodels as bsm

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
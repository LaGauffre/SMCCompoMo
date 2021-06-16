# -*- coding: utf-8 -*-
# +
"""
Created on Fri Feb 26 20:16:24 2021

@author: pierr
"""

# import os
# for env in ["MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "OMP_NUM_THREADS"]:
#     os.environ[env] = "1"

from smclomo import *

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
import pymc3 as pm
import matplotlib.pyplot as plt
from time import time
from IPython.display import display_html


# -


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

# Graphic methods
def model_grid(df, metric):
    df["w"] = np.exp(df[metric] - np.max(df[metric])) / np.sum(np.exp(df[metric] - np.max(df[metric]))) 
    criterium = "w"

    # Draw each cell as a scatter point with varying size and color
    g = sns.relplot( data=df,
        x="Body", y="Tail", size=criterium, color = "black",
        height=5, sizes=(25, 500), size_norm=(0,1), aspect = 1
    )
    g._legend.remove()
    # # Tweak the figure to finalize
    g.set(xlabel="Body", ylabel="Tail", aspect="equal")
    g.despine(left=True, bottom=True)
    # g.ax.margins(.02)
    for label in g.ax.get_xticklabels():
        label.set_rotation(90)
    for artist in g.legend.legendHandles:
        artist.set_edgecolor(".7")
    # title
    fig.tight_layout()
    new_title = 'Model weights'
    g._legend.set_title(new_title)
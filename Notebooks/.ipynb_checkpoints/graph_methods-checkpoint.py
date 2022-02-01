import scipy as sc
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def model_grid(df, metric, x=0.05, y=0.05):
    df["w"] = np.exp(df[metric] - np.max(df[metric])) / np.sum(np.exp(df[metric] - np.max(df[metric]))) 
    criterium = "w"
 
    # Draw each cell as a scatter point with varying size and color
    g = sns.relplot( data=df,
        x="Body", y="Tail",  size=criterium, color = "#006699",
        height=5, sizes=(25, 400), size_norm=(0,1), aspect = 1
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
    xlim = g.ax.get_xlim()
    ylim = g.ax.get_ylim()

    xmargin = (xlim[1]-xlim[0])*x
    ymargin = (ylim[1]-ylim[0])*y

    g.ax.set_xlim(xlim[0]-xmargin,xlim[1]+xmargin)
    g.ax.set_ylim(ylim[0]-ymargin,ylim[1]+ymargin)
    # title
    # fig.tight_layout()
    new_title = 'Model weights'
    g._legend.set_title(new_title)
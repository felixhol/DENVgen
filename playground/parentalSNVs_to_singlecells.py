# vim: fdm=indent
'''
author:     Fabio Zanini
date:       10/12/18
content:    Check fate of parental SNV frequencies.
'''
import os
import sys
import pysam

import numpy as np
import pandas as pd
import xarray as xr
from collections import Counter, defaultdict
from Bio import SeqIO
import matplotlib.pyplot as plt
import seaborn as sns


if __name__ == '__main__':

    data_par = xr.open_dataset('../bigdata/allele_frequencies_parental.nc')
    data_sc = xr.open_dataset('../bigdata/allele_frequencies.nc')

    af_par = data_par['aaf_avg']
    ma_par = data_par['alt']

    # Highest frequencies
    poss = af_par.fillna(0).data.argsort()[::-1][:50]

    df = data_sc['aaf'][poss].to_dataframe()
    def logit(x, ext=1e-3):
        return -np.log10(1. / np.minimum(np.maximum(x, ext), 1 - ext) - 1)
    df = logit(df)
    df['position'] = df.index.get_level_values('position')
    df.index = np.arange(df.shape[0])
    df = df.loc[~np.isnan(df['aaf'])]
    fig, ax = plt.subplots(figsize=(22, 3))
    sns.violinplot(
            data=df,
            x='position',
            y='aaf',
            ax=ax,
            scale='width',
            order=poss,
            )
    for x, tk in zip(ax.get_xticks(), ax.get_xticklabels()):
        pos = int(tk.get_text())
        y = logit(af_par[pos])
        ax.scatter([x - 0.1], [y], s=30, color='red', zorder=5)
    ax.set_xlim(-0.5, len(poss) - 0.5)

    plt.tight_layout()
    plt.ion()
    plt.show()

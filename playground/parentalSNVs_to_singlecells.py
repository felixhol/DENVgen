# vim: fdm=indent
'''
author:     Fabio Zanini
date:       10/12/18
content:    Check fate of parental SNV frequencies.
'''
import os
import sys
import pysam
import pickle
import numpy as np
import pandas as pd
import xarray as xr
from collections import Counter, defaultdict
from Bio import SeqIO
import matplotlib.pyplot as plt
import seaborn as sns

os.environ['SINGLET_CONFIG_FILENAME'] = 'singlet.yml'
sys.path.append('/home/fabio/university/postdoc/singlet')
from singlet.dataset import Dataset, CountsTable


if __name__ == '__main__':

    data_par = xr.open_dataset('../bigdata/allele_frequencies_parental.nc')
    data_sc = xr.open_dataset('../bigdata/allele_frequencies.nc')

    af_par = data_par['aaf_avg']
    ma_par = data_par['alt']
    afs = data_sc['aaf']

    # Highest frequencies
    poss_par = af_par.fillna(0).data.argsort()[::-1]
    poss_sc = (afs * (1 - afs)).mean(axis=1).fillna(0).data.argsort()[::-1]
    poss = np.union1d(poss_par[:20], poss_sc[:20])

    ## Scatter ranks in parental versus rank in single cells
    #ranks_par = np.empty_like(poss_par)
    #ranks_par[poss_par] = np.arange(len(poss_par))
    #ranks_sc = np.empty_like(poss_sc)
    #ranks_sc[poss_sc] = np.arange(len(poss_sc))
    #fig, ax = plt.subplots(figsize=(4, 3.5))
    #ax.scatter(ranks_par, ranks_sc, s=10, alpha=0.7)
    #ax.set_xlabel('ranks in parental')
    #ax.set_ylabel('ranks in single cells')
    #plt.tight_layout()

    # Plot distribution of af in single cells
    df = afs[poss].to_dataframe()
    def logit(x, ext=1e-3):
        return -np.log10(1. / np.minimum(np.maximum(x, ext), 1 - ext) - 1)
    df = logit(df)
    df['position'] = df.index.get_level_values('position')
    df.index = np.arange(df.shape[0])
    df = df.loc[~np.isnan(df['aaf'])]
    df['aaf'].clip(lower=1e-3, upper=1-1e-3, inplace=True)
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
    ax.set_ylabel('$\\nu$', fontsize=14)
    yticklabels = ['0.001', '0.01', '0.1', '0.5', '0.9', '0.99', '0.999']
    ax.set_yticklabels(yticklabels)
    ax.set_yticks(logit([float(x) for x in yticklabels]))
    plt.tight_layout()

    # Load SNV cluster data
    with open('../data/metadataD_SNV_with_tsne_and_tsneSNV.pkl', 'rb') as ff:
        metadata_felix = pickle.load(ff)
    clusters = metadata_felix['clusterN_SNV'].fillna('none')

    # Same plot but by SNV cluster
    df = afs[poss].to_dataframe()
    df = logit(df)
    df['position'] = df.index.get_level_values('position')
    df['clusterN'] = clusters.loc[df.index.get_level_values('sample')].values
    df.index = np.arange(df.shape[0])
    df = df.loc[~np.isnan(df['aaf'])]
    #df['aaf'].clip(lower=1e-3, upper=1-1e-3, inplace=True)
    fig, axs = plt.subplots(2, 1, figsize=(17, 5.5))
    for iax, ax in enumerate(axs):
        if iax == 0:
            posi = poss[:(len(poss) // 2)]
        else:
            posi = poss[(len(poss) // 2):]
        dfi = df.loc[df['position'].isin(posi)]
        sns.violinplot(
                data=dfi,
                x='position',
                hue='clusterN',
                y='aaf',
                ax=ax,
                scale='width',
                order=posi,
                hue_order=[0, 1, 2, 3, 4, 5, 'none'],
                cut=0,
                lw=0.5,
                )
        for x, tk in zip(ax.get_xticks(), ax.get_xticklabels()):
            pos = int(tk.get_text())
            y = logit(af_par[pos])
            ax.plot([x - 0.5, x + 0.5], [y, y], lw=1.5, color='red', zorder=5)
        ax.set_xlim(-0.5, len(posi) - 0.5)
        ax.set_ylim(-3.1, 3.1)
        ax.set_ylabel('$\\nu$', fontsize=14)
        yticklabels = ['0.001', '0.01', '0.1', '0.5', '0.9', '0.99', '0.999']
        ax.set_yticklabels(yticklabels)
        ax.set_yticks(logit([float(x) for x in yticklabels]))
        ax.legend(loc='upper right', ncol=3, title='cluster', fontsize=10)
        for x in 0.5 *(ax.get_xticks()[:-1] + ax.get_xticks()[1:]):
            ax.plot([x, x], logit(np.array([1e-10, 1-1e-10]), ext=1e-4), lw=1, color='grey', alpha=0.5)
        pcs = ax.get_children()
        for pc in pcs:
            if isinstance(pc, type(pcs[0])):
                pc.set_linewidth([1])
        if iax != (len(axs) - 1):
            ax.legend_.remove()
    plt.tight_layout()

    # Calculate Hamming distance distributions between clusters
    # FIXME: missing values??
    from scipy.spatial.distance import cdist, squareform
    clustersu = [0, 1, 2, 3, 4, 5, 'none']
    dclu = {}
    for i1, c1 in enumerate(clustersu):
        s1 = clusters.index[clusters == c1]
        af1 = afs.loc[:, s1].fillna(0).data.T
        for c2 in clustersu[:i1+1]:
            s2 = clusters.index[clusters == c2]
            af2 = afs.loc[:, s2].fillna(0).data.T
            if c1 != c2:
                d = cdist(af1, af2).ravel()
            else:
                d = squareform(cdist(af1, af2))
            dclu[frozenset([c1, c2])] = d

    # Calculate averages and hierarchical clustering
    from scipy.cluster.hierarchy import linkage, leaves_list
    dcluavg = {key: d.mean() for key, d in dclu.items()}
    dclumat = np.zeros((len(clustersu), len(clustersu)))
    for i1, c1 in enumerate(clustersu):
        s1 = clusters.index[clusters == c1]
        af1 = afs.loc[:, s1].fillna(0).data.mean(axis=1)
        for i2, c2 in enumerate(clustersu[:i1+1]):
            s2 = clusters.index[clusters == c2]
            af2 = afs.loc[:, s2].fillna(0).data.mean(axis=1)
            dclumat[i1, i2] = dclumat[i2, i1] = np.sqrt(((af1 - af2)**2).sum())
    z = linkage(squareform(dclumat), optimal_ordering=True)
    indu = leaves_list(z)

    # Plot cluster Hamming distances
    fig, axs = plt.subplots(
        len(clustersu), len(clustersu),
        figsize=(9, 8.3),
        sharex=True,
        sharey=True,
        )
    colors = sns.color_palette(n_colors=len(clustersu))
    for iax1 in range(len(clustersu)):
        for iax2 in range(len(clustersu)):
            c1 = clustersu[indu[iax1]]
            c2 = clustersu[indu[iax2]]
            ax = axs[iax1, iax2]
            d = dclu[frozenset([c1, c2])]
            sns.kdeplot(d, ax=ax, shade=True, bw=0.2, color='grey')
            if iax2 == 0:
                ax.set_ylabel(c1)
            if iax1 == len(clustersu) - 1:
                ax.set_xlabel(c2)
            if iax1 == iax2:
                ax.set_facecolor(list(colors[indu[iax1]]) + [0.2])
            ax.grid(True)
            ax.set_xlim(0, 3.9)
            ax.set_xticks([0, 1, 2, 3, 4])
            ax.set_xticks([0.5, 1.5, 2.5, 3.5], minor=True)
            ax.set_yticklabels([])
    fig.text(0.52, 0.02, 'cluster #', ha='center')
    fig.text(0.02, 0.52, 'cluster #', va='center', rotation=90)
    fig.suptitle('Hamming distance distributions across SNV clusters')
    plt.tight_layout(h_pad=0, w_pad=0, rect=(0.03, 0.03, 1, 0.97))

    # Calculate transciptome distances
    ds = Dataset(
            samplesheet='dengue',
            counts_table='dengue',
            featuresheet='humanGC38',
            )
    ds.samplesheet['cluster_SNV'] = clusters
    ds.counts.normalize(inplace=True)
    ds.rename(axis='features', column='GeneName', inplace=True)
    ds.feature_selection.unique(inplace=True)

    # Restrict to differentially expresse genes
    with open('../data/genes_diff_expressed_clustersSNV.tsv', 'rt') as f:
        genes = f.read().split('\t')
    dsd = ds.query_features_by_name(genes)
    dsd.counts.log(inplace=True)

    dsp = dsd.split('cluster_SNV')
    dclut = {}
    for i1, c1 in enumerate(clustersu):
        ge1 = dsp[c1].counts.values.T
        for c2 in clustersu[:i1+1]:
            print(c1, c2)
            ge2 = dsp[c2].counts.values.T
            if c1 != c2:
                d = cdist(ge1, ge2).ravel()
            else:
                d = squareform(cdist(ge1, ge2))
            dclut[frozenset([c1, c2])] = d

    # Plot
    fig, axs = plt.subplots(
        len(clustersu), len(clustersu),
        figsize=(9, 8.3),
        sharex=True,
        sharey=True,
        )
    colors = sns.color_palette(n_colors=len(clustersu))
    for iax1 in range(len(clustersu)):
        for iax2 in range(len(clustersu)):
            c1 = clustersu[indu[iax1]]
            c2 = clustersu[indu[iax2]]
            ax = axs[iax1, iax2]
            d = dclut[frozenset([c1, c2])]
            sns.kdeplot(d, ax=ax, shade=True, bw=2, color='grey')
            if iax2 == 0:
                ax.set_ylabel(c1)
            if iax1 == len(clustersu) - 1:
                ax.set_xlabel(c2)
            if iax1 == iax2:
                ax.set_facecolor(list(colors[indu[iax1]]) + [0.2])
            ax.grid(True)
            ax.set_xlim(0, 30)
            ax.set_yticklabels([])
    fig.text(0.52, 0.02, 'cluster #', ha='center')
    fig.text(0.02, 0.52, 'cluster #', va='center', rotation=90)
    fig.suptitle('Euclidean transcriptome distance distributions across SNV clusters')
    plt.tight_layout(h_pad=0, w_pad=0, rect=(0.03, 0.03, 1, 0.97))



    plt.ion()
    plt.show()

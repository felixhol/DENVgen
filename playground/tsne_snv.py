# vim: fdm=indent
'''
author:     Fabio Zanini
date:       29/11/18
content:    
'''
# Modules
import os
import sys
import pickle
import argparse
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

os.environ['SINGLET_CONFIG_FILENAME'] = 'singlet.yml'
sys.path.append('/home/fabio/university/postdoc/singlet')
from singlet.dataset import Dataset, CountsTable


# Script
if __name__ == '__main__':

    ds = Dataset(
            samplesheet='dengue',
            )
    data = xr.open_dataset('../bigdata/allele_frequencies.nc')
    ds.counts = CountsTable(data['aaf'].to_pandas().fillna(0))

    # Sync with Felix metadata
    with open('../data/metadataD_SNV_with_tsne.pkl', 'rb') as ff:
        metadata_felix = pickle.load(ff)
    samples = metadata_felix.index[(~np.isnan(metadata_felix[['Dn', 'Ds']])).all(axis=1)]
    ds.samplesheet = ds.samplesheet.loc[samples]
    metadata_felix = metadata_felix.loc[samples]
    for col in ['coverage', 'Ds', 'Dn', 'depth', 'numSNV', 'Dn_s', 'tsne1_MOI1_10', 'tsne2_MOI1_10']:
        ds.samplesheet[col] = metadata_felix[col]
    ds.samplesheet['log_Dn'] = np.log10(1e-6 + ds.samplesheet['Dn'])
    ds.samplesheet['log_Ds'] = np.log10(1e-6 + ds.samplesheet['Ds'])

    cov = ds.samplesheet['coverage']
    n = ds.samplesheet['numberDengueReads'].astype(int)
    ds.samplesheet['virus_reads_per_million'] = 1e6 * n / (cov + n)
    ds.samplesheet['log_virus_reads_per_million'] = np.log10(0.1 + 1e6 * n / (cov + n))

    vs = ds.samplesheet[['tsne1_MOI1_10', 'tsne2_MOI1_10']]

    # Restrict to high variance SNVs
    dsv = ds.copy()
    ind = ds.counts.values.var(axis=1).argsort()[-200:]
    dsv.counts = dsv.counts.iloc[ind]
    vsg = dsv.dimensionality.tsne(perplexity=30)

    # Cluster
    # NOTE: the number is manually chosen but does not matter much atm
    dsv.samplesheet['clusterN'] = ds.cluster.kmeans(axis='samples', n_clusters=6)
    dss = dsv.split('clusterN')

    # Plot gene expression tSNE overlayed with viral genomics
    fig, axs = plt.subplots(3, 3, figsize=(7, 6), sharex=True, sharey=True)
    axs = axs.ravel()
    plots = ['log_virus_reads_per_million', 'coverage', 'depth', 'numSNV', 'log_Dn', 'log_Ds', 'time [h]', 'MOI', 'clusterN']
    for ax, col in zip(axs, plots):
        dsv.plot.scatter_reduced_samples(vsg, ax=ax, color_by=col, s=5)
        ax.set_axis_off()
        ax.set_title(col, fontsize=10)
    for ic, dsi in dss.items():
        x = vsg.loc[dsi.samplenames, 'dimension 1'].values.mean()
        y = vsg.loc[dsi.samplenames, 'dimension 2'].values.mean()
        ax.text(x, y, str(ic), fontsize=10, ha='center', va='center')
    plt.tight_layout(h_pad=0, w_pad=0)

    # Identify marker SNVs for the clusters
    # Cluster 0 is the largest, by definition
    snvs = {}
    for i in range(1, 6):
        comp = dss[i].compare(dss[0])
        snvs[i] = comp.nsmallest(n=10, columns='P-value').index.values
    snv_posall = np.unique(np.concatenate(list(snvs.values())))

    # Plot distributions of SNVs in the clusters
    df = dsv.counts.loc[snv_posall].T

    # Logit transform (it's hard to estimate those violins otherwise)
    def logit(x, ext=1e-3):
        return -np.log10(1. / np.minimum(np.maximum(x, ext), 1 - ext) - 1)
    df = logit(df)
    yticklabels = np.array(['0.01', '0.1', '0.5', '0.9', '0.99'])
    yticks = logit([float(x) for x in yticklabels])

    df['clusterN'] = dsv.samplesheet['clusterN']
    fig, axs = plt.subplots(2, 7, figsize=(15, 4.5), sharex=True, sharey=True)
    axs = axs.ravel()
    for iax, (pos, ax) in enumerate(zip(snv_posall, axs)):
        sns.swarmplot(
                data=df[[pos, 'clusterN']],
                ax=ax,
                x='clusterN',
                y=pos,
                hue='clusterN',
                #legend=False,
                #scale='width',
                #bw=1,
                )
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_title(str(pos))
        ax.grid(True, axis='y')
        ax.legend_.remove()
        ax.set_ylim(-3.2, 3.2)
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels)
    fig.text(0.52, 0.02, 'cluster number', ha='center')
    fig.text(0.02, 0.52, '$\\nu$', ha='center', va='center', rotation=90)
    plt.tight_layout(h_pad=0.5, w_pad=0, rect=(0.03, 0.03, 1, 1))

    # Save tSNE coordinates on SNVs and clusterN in Felix's pickle
    with open('../data/metadataD_SNV_with_tsne.pkl', 'rb') as ff:
        metadata_felix = pickle.load(ff)
    metadata_felix['tsne1_SNV'] = np.nan
    metadata_felix.loc[vsg.index, 'tsne1_SNV'] = vsg['dimension 1']
    metadata_felix['tsne2_SNV'] = np.nan
    metadata_felix.loc[vsg.index, 'tsne2_SNV'] = vsg['dimension 2']
    metadata_felix['clusterN_SNV'] = np.nan
    metadata_felix.loc[dsv.samplenames, 'clusterN_SNV'] = dsv.samplesheet['clusterN']
    with open('../data/metadataD_SNV_with_tsne_and_tsneSNV.pkl', 'wb') as ff:
        pickle.dump(metadata_felix, ff, protocol=-1)


    plt.ion()
    plt.show()

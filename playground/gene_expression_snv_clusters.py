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
            counts_table='dengue',
            featuresheet='humanGC38',
            )
    data_snv = xr.open_dataset('../bigdata/allele_frequencies.nc')
    #ds.counts = CountsTable(data['aaf'].to_pandas().fillna(0))

    # Sync with Felix metadata
    with open('../data/metadataD_SNV_with_tsne_and_tsneSNV.pkl', 'rb') as ff:
        metadata_felix = pickle.load(ff)
    samples = metadata_felix.index[(~np.isnan(metadata_felix[['Dn', 'Ds']])).all(axis=1)]
    ds.samplesheet = ds.samplesheet.loc[samples]
    metadata_felix = metadata_felix.loc[samples]
    for col in ['coverage', 'Ds', 'Dn', 'depth', 'numSNV', 'Dn_s', 'tsne1_MOI1_10', 'tsne2_MOI1_10',
                'tsne1_SNV', 'tsne2_SNV', 'clusterN_SNV']:
        ds.samplesheet[col] = metadata_felix[col]
    ds.samplesheet['log_Dn'] = np.log10(1e-6 + ds.samplesheet['Dn'])
    ds.samplesheet['log_Ds'] = np.log10(1e-6 + ds.samplesheet['Ds'])

    cov = ds.samplesheet['coverage']
    n = ds.samplesheet['numberDengueReads'].astype(int)
    ds.samplesheet['virus_reads_per_million'] = 1e6 * n / (cov + n)
    ds.samplesheet['log_virus_reads_per_million'] = np.log10(0.1 + 1e6 * n / (cov + n))

    vs = ds.samplesheet[['tsne1_MOI1_10', 'tsne2_MOI1_10']]
    vsg = ds.samplesheet[['tsne1_SNV', 'tsne2_SNV']]

    # Translate gene IDs
    ds.rename(axis='features', column='GeneName', inplace=True)
    ds.feature_selection.unique(inplace=True)

    ## Restrict to high variance SNVs
    #dsv = ds.copy()
    #ind = ds.counts.values.var(axis=1).argsort()[-200:]
    #dsv.counts = dsv.counts.iloc[ind]

    # Find upregulated genes
    clusters = np.unique(ds.samplesheet['clusterN_SNV'])
    genes = {}
    for ic in clusters:
        ds.samplesheet['clusterN_SNV_{:}'.format(ic)] = ds.samplesheet['clusterN_SNV'] == ic
        dss = ds.split('clusterN_SNV_{:}'.format(ic))
        comp = dss[True].compare(dss[False])

        # FIXME: maybe look symmetrically for up- and downregulated
        comp['diff'] = dss[True].counts.mean(axis=1) - dss[False].counts.mean(axis=1)
        genesi = comp.loc[comp['diff'] > 0, 'P-value'].nsmallest(n=5).index.values

        genes[ic] = genesi

    genes_all = np.unique(np.concatenate(list(genes.values())))

    dsv = ds.query_features_by_name(genes_all)

    # Plot distributions
    fig, axs = plt.subplots(3, 10, figsize=(17, 7), sharex=True, sharey=True)
    axs = axs.ravel()
    for ax, gene in zip(axs, genes_all):
        df = np.log10(0.1 + dsv.counts.loc[[gene]].T)
        df['clusterN'] = dsv.samplesheet['clusterN_SNV']
        sns.boxplot(
                data=df,
                y=gene,
                x='clusterN',
                ax=ax,
                order=clusters,
                )
        ax.grid(axis='y')
        ax.set_title(gene)
        ax.set_ylabel('')
        ax.set_xlabel('')
        ax.set_ylim(-1.1, 5.1)
        for ic in clusters:
            if gene in genes[ic]:
                ax.text(ic, 5, '*', fontsize=10, va='top', ha='center')
    fig.text(0.52, 0.02, 'cluster #', ha='center', va='bottom')
    fig.text(0.02, 0.52, 'log10 (cpm)', ha='left', va='center', rotation=90)
    plt.tight_layout(w_pad=0, rect=(0.03, 0.03, 1, 1))

    fig, ax = plt.subplots(figsize=(2.4, 2.6))
    df = dsv.samplesheet[['clusterN_SNV', 'log_virus_reads_per_million']]
    sns.boxplot(
            data=df,
            y='log_virus_reads_per_million',
            x='clusterN_SNV',
            ax=ax,
            order=clusters,
            )
    ax.grid(axis='y')
    ax.set_title('vRNA abundance')
    ax.set_xlabel('cluster #')
    ax.set_ylabel('log10 (cpm)')
    ax.set_ylim(-1.1, 5.1)
    plt.tight_layout()


    plt.ion()
    plt.show()


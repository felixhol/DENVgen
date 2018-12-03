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
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

os.environ['SINGLET_CONFIG_FILENAME'] = 'singlet.yml'
sys.path.append('/home/fabio/university/postdoc/singlet')
from singlet.dataset import Dataset


# Script
if __name__ == '__main__':

    ds = Dataset(
            samplesheet='dengue',
            counts_table='dengue',
            featuresheet='humanGC38',
            )

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

    # Plot gene expression tSNE overlayed with viral genomics
    fig, axs = plt.subplots(3, 3, figsize=(7, 6))
    axs = axs.ravel()
    plots = ['log_virus_reads_per_million', 'coverage', 'depth', 'numSNV', 'log_Dn', 'log_Ds', 'time [h]', 'MOI', 'Dn_s']
    for ax, col in zip(axs, plots):
        ds.plot.scatter_reduced_samples(vs, ax=ax, color_by=col, s=5)
        ax.set_axis_off()
        ax.set_title(col, fontsize=10)
    plt.tight_layout(h_pad=0, w_pad=0)

    # Correlate viral genomics with itself
    corr_pheno = ds.correlation.correlate_phenotypes_phenotypes(
        ['log_virus_reads_per_million', 'coverage', 'depth',
         'numSNV', 'Dn', 'Ds', 'Dn_s'])

    # Correlate viral genomics with gene expression
    ds.rename(axis='features', column='GeneName', inplace=True)
    ds.feature_selection.unique(inplace=True)

    # Nothing is strongly correlated or anticorrelated, but NOL3 might be an
    # interesting anticorrelated hit


    plt.ion()
    plt.show()

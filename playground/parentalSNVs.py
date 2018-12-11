# vim: fdm=indent
'''
author:     Fabio Zanini
date:       10/12/18
content:    Consolidate parental SNV frequencies from VCF files.
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

    # Example file
    #fdn = '../bigdata/DENVparentalVCF'
    #fn_ex = 'varsDENV_amp1_d025.vcf'
    #f = pysam.VariantFile('{:}/{:}'.format(fdn, fn_ex), 'r')
    #poss = defaultdict(list)
    #for line in f:
    #    poss[line.pos].append(line)

    # Prepare output data
    seq = SeqIO.read('../data/U87411_strain16681.gb', 'gb')
    positions = np.arange(len(seq))
    alpha = np.array(['A', 'C', 'G', 'T', '-', 'N'])
    alphal = alpha.tolist()
    # There are 4 amplicons
    afs = np.zeros((len(alpha), len(positions), 4))
    cov = np.zeros((len(positions), 4), int)

    # Default to the reference
    for ipos, a in enumerate(seq):
        ia = alphal.index(a)
        afs[ia, ipos, :] = 1

    # Fill the output
    fdn = '../bigdata/DENVparentalVCF'
    for fn in os.listdir(fdn):
        isn = int(fn.split('_')[1][-1]) - 1
        with pysam.VariantFile('{:}/{:}'.format(fdn, fn), 'r') as f:
            for line in f:
                ipos = line.pos - 1
                cov[ipos, isn] = line.info['DP']
                if len(line.alts) == 1:
                    aff = [line.info['AF']]
                else:
                    aff = line.info['AF']
                iaa = alphal.index(line.ref)
                for a, af in zip(line.alts, aff):
                    ia = alphal.index(a)
                    afs[ia, ipos, isn] = af
                    afs[iaa, ipos, isn] -= af

    # Calculate alternative allele frequencies
    maf = np.ma.masked_all((len(seq), 4))
    ma = np.ma.masked_all(len(seq), '<U1')
    for ipos in range(len(seq)):
        afp = afs[:, ipos, :].max(axis=1)
        mai = afp.argsort()[-2]
        if afp[mai] > 0:
            ma[ipos] = alpha[mai]
            maf[ipos] = afs[mai, ipos]
    maf = xr.DataArray(
        data=maf,
        dims=['position', 'amplicon'],
        coords={'position': positions, 'amplicon': [1, 2, 3, 4]},
        )
    afs = xr.DataArray(
        data=afs,
        dims=['allele', 'position', 'amplicon'],
        coords={'allele': alpha, 'position': positions, 'amplicon': [1, 2, 3, 4]},
        )
    cov = xr.DataArray(
        data=cov,
        dims=['position', 'amplicon'],
        coords={'position': positions, 'amplicon': [1, 2, 3, 4]},
        )

    # Trusted regions (no extremes, no primers, manual annotation)
    trusted = [[0, 3531], [2645, 5995], [5021, 8476], [7475, 10540]]
    for iaf in range(4):
        maf[:trusted[iaf][0], iaf] = np.nan
        maf[trusted[iaf][1]:, iaf] = np.nan

    # Average in overlaps (it looks good)
    maf_avg = maf.mean(axis=1)

    data = xr.Dataset(data_vars={
        'afs': afs,
        'depth': cov,
        'ref': np.array(seq),
        'alt': ma,
        'aaf': maf,
        'aaf_avg': maf_avg,
        },
        attrs={
            'afs': 'allele frequencies',
            'depth': 'sequencing depth at polymorphic sites',
            'ref': 'reference alleles',
            'alt': 'top alternative allele',
            'aaf': 'frequency of top alternative allele (if present)',
            'aaf_avg': 'like aaf, but averaged in amplicon overlaps',
            })

    # NetCDF is large but fast to I/O
    data.to_netcdf(path='../bigdata/allele_frequencies_parental.nc', format='NETCDF4')
    # To open back the allele frequencies:
    # import xarray as xr
    # a = xr.open_dataset('../bigdata/allele_frequencies_parental.nc')

    # Plot minor allele frequencies
    fig, axs = plt.subplots(1, 2, figsize=(17, 8), gridspec_kw={'width_ratios': [20, 1]}, sharey=True)
    ax = axs[0]
    x = maf['position'].data
    colors = sns.color_palette(n_colors=4)
    for iamp in range(4):
        y = maf.loc[:, iamp + 1].fillna(-1).data
        ind = y > 0
        yp = np.minimum(np.maximum(y[ind], 1e-3), 1 - 1e-3)
        xp = x[ind]
        ax.scatter(
            xp, yp,
            s=10, alpha=0.7, color=colors[iamp], label=str(iamp+1),
            zorder=1,
            )
    # Plot lines in the overlaps
    pos_overlap = ((maf.fillna(-1).data >= 0).sum(axis=1) == 2).nonzero()[0]
    diffs_overlap = [[], [], []]
    for pos in pos_overlap:
        afp = maf.fillna(-1).data[pos]
        indy = (afp >= 0).nonzero()[0]
        y = np.minimum(np.maximum(afp[indy], 1e-3), 1 - 1e-3)
        x = [pos] * len(y)
        diffs_overlap[indy.min()].append(np.abs(y[0] - y[1]))
        ax.plot(x, y, ls='-', color='grey', alpha=0.5, zorder=0)

    ax.grid(True)
    ax.legend(loc='upper right', title='Amplicon:', ncol=2)
    ax.set_ylim(0.9e-2, 1 - 0.9e-2)
    ax.set_yscale('logit')
    ax.set_xlabel('Position in DENV genome [bp]')
    ax.set_ylabel('$\\nu$', fontsize=14)

    ax = axs[1]
    bins = np.logspace(-3, 0, 31)
    for i in range(3):
        h = np.histogram(diffs_overlap[i], bins=bins, density=True)[0]
        y = np.sqrt(bins[1:] * bins[:-1])
        x = h + 0.09
        ax.plot(x, y, lw=2, color=colors[i])
    ax.set_xlim(left=0.1)
    ax.set_xscale('log')
    ax.set_xlabel('Density')
    ax.set_ylabel('$\\vert \\nu_1 - \\nu_2 \\vert$', fontsize=14)
    ax.grid(True, axis='x')

    plt.tight_layout()


    plt.ion()
    plt.show()


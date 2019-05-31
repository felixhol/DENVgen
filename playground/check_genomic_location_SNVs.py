# vim: fdm=indent
'''
author:     Fabio Zanini
date:       11/03/19
content:    Check the genomic location of the SNVs, especially parental ones,
            to try and get at function.
'''
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


def plot_snv_frequency_distributions(afs, poss, af_par):
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

    return {'fig': fig, 'ax': ax}


# Script
if __name__ == '__main__':

    data_par = xr.open_dataset('../bigdata/allele_frequencies_parental.nc')
    data_sc = xr.open_dataset('../bigdata/allele_frequencies.nc')

    af_par = data_par['aaf_avg']
    ma_par = data_par['alt']
    afs = data_sc['aaf']

    # Select by highest frequencies and highest variance
    poss_par = af_par.fillna(0).data.argsort()[::-1]
    poss_sc = (afs * (1 - afs)).mean(axis=1).fillna(0).data.argsort()[::-1]
    poss_all = np.union1d(poss_par[:20], poss_sc[:20])

    # Plot SNV frequencies in single cells and in the parental strain
    #d = plot_snv_frequency_distributions(afs, poss, af_par)

    # Manually select a few positions from earlier plots, they are are subset
    # of poss_all. We could use all, but this is a good list for now
    poss = np.array([
        398, 401, 1530, 2280, 2424, 2425, 2426, 2429, 4828,
        5046, 5558, 5561, 8554, 8577,
        ])

    # Load annotated reference
    from Bio import SeqIO
    from Bio.Seq import translate
    ref_fn = '../data/U87411_strain16681.gb'
    ref = SeqIO.read(ref_fn, format='gb')
    # print peptides
    #for fea in ref.features:
    #    print(fea)

    # Format the changes
    # NOTE: pos 96 (in 0-based coordinates) is the start of the CDS, setting the
    # reading frame. Stop codon is 10269 - 10272
    start = 96
    mutations = []
    for pos in poss:
        pos = int(pos)  # NOTE: Biopython bug
        mat = data_sc['afs'].sel(position=pos)
        # Alleles
        alleles = mat['allele'][mat.mean(axis=1).data.argsort()[::-1]][:2].data
        # Reference
        refa = data_par['ref'][pos].data.tolist()
        # Mutation
        alt = list(set(alleles) - set([refa]))[0]
        # Context
        context_RNA = str(ref.seq[pos - 10: pos])+ref[pos].lower()+str(ref.seq[pos+1: pos+10])
        # Context protein
        codon_pos = (pos - start) % 3
        codon_pos_start = pos - codon_pos
        codon_pos_end = codon_pos_start + 3
        codon_ref = str(ref[codon_pos_start: codon_pos_end].seq)
        aa_ref = translate(codon_ref)
        codon_alt = codon_ref[:codon_pos] + alt + codon_ref[codon_pos + 1:]
        aa_alt = translate(codon_alt)
        context_protein = \
                translate(str(ref[codon_pos_start - 12: codon_pos_start].seq)) + \
                aa_ref.lower() + \
                translate(str(ref[codon_pos_end: codon_pos_end + 12].seq))
        syn = 'syn' if aa_ref == aa_alt else 'nonsyn'
        aa_change = '{:} > {:}'.format(aa_ref, aa_alt) if aa_ref != aa_alt else ''

        mut = {
            'position': pos,
            'position_1based': pos + 1,
            'ref': refa,
            'alt': alt,
            'change': '{:} > {:}'.format(refa, alt),
            'context_RNA': context_RNA,
            'aa_ref': aa_ref,
            'aa_alt': aa_alt,
            'aa_change': aa_change,
            'syn': syn,
            'context_protein': context_protein,
            }
        mut['id'] = str(pos+1)+' '+mut['change']
        mutations.append(mut)
    mutations = pd.DataFrame(mutations).set_index('id')
    #mutations.to_csv('../data/mutations_highvariance_summary.tsv', sep='\t', index=True)

    print('Combine with transcriptomics')
    # FIXME: this is actually done in the rnavelocity.py script

    print('Load RNA velocity results onto tSNE embedding')
    vse = pd.read_csv('../data/velocity_tsne_projection.tsv', sep='\t', index_col=0)
    vs = vse.iloc[:, :2].values
    dvs = vse.iloc[:, 2:].values
    index = vse.index


    # This is transcriptomics stuff
    if False:
        ds = Dataset(
                samplesheet='dengue',
                counts_table='dengue',
                featuresheet='humanGC38',
                )
    
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
        ds.counts.normalize(inplace=True)
    
    

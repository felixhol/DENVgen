# vim: fdm=indent
'''
author:     Fabio Zanini
date:       19/12/18
content:    Try RNA velocity
'''
import os
import sys
import pickle
import numpy as np
import pandas as pd
import velocyto as vcy
import matplotlib.pyplot as plt
import seaborn as sns

os.environ['SINGLET_CONFIG_FILENAME'] = 'singlet.yml'
sys.path.append('/home/fabio/university/postdoc/singlet')
from singlet.dataset import Dataset


if __name__ == '__main__':

    # velocyto skips quasi-empty BAM files, final order is lexicographic as in:
    # echo '' > ~/subfolders.tsv; for fdn in 10017006*; do si=$(du $fdn/star/Aligned.out.possorted.bam | cut -f1); if [ $si -ge "30" ]; then echo $fdn >> ~/subfolders.tsv; fi; done
    cellnames = pd.read_csv('../bigdata/rnavelocity_cellnames.tsv', header=None).values[:, 0]
    vlm = vcy.VelocytoLoom("../bigdata/rna_velocity.loom")
    vlm.ca['CellID'] = cellnames

    # Load and sync external metadata
    ds = Dataset(
            #counts_table='dengue',
            samplesheet='dengue',
            )
    with open('../data/metadataD_SNV_with_tsne_and_tsneSNV.pkl', 'rb') as ff:
        metadata_felix = pickle.load(ff)
    ds.samplesheet = ds.samplesheet.loc[cellnames]
    metadata_felix = metadata_felix.loc[cellnames]

    vlm.ca['ClusterName'] = metadata_felix['clusterN_SNV'].fillna(6).values
    vlm.set_clusters(vlm.ca["ClusterName"])

    ds.samplesheet['clusterN_SNV'] = metadata_felix['clusterN_SNV'].fillna(6)
    ds.samplesheet['coverage'] = metadata_felix['coverage']
    ds.samplesheet['virus_reads_per_million'] = 1.0 * 1e6 * ds.samplesheet['numberDengueReads'] / (ds.samplesheet['numberDengueReads'] + ds.samplesheet['coverage'])
    ds.samplesheet['log_virus_reads_per_million'] = np.log10(0.1 + ds.samplesheet['virus_reads_per_million'])

    vlm.normalize("S", size=True, log=True)
    vlm.filter_cells(bool_array=vlm.initial_Ucell_size > np.percentile(vlm.initial_Ucell_size, 0.5))
    ds.samplesheet = ds.samplesheet.loc[vlm.ca['CellID']]
    metadata_felix = metadata_felix.loc[vlm.ca['CellID']]

    vlm.score_detection_levels(min_expr_counts=40, min_cells_express=30)
    vlm.filter_genes(by_detection_levels=True)
    vlm.score_cv_vs_mean(3000, plot=True, max_expr_avg=35)
    vlm.filter_genes(by_cv_vs_mean=True)

    vlm._normalize_S(relative_size=vlm.S.sum(0),
        target_size=vlm.S.sum(0).mean())
    vlm._normalize_U(relative_size=vlm.U.sum(0),
        target_size=vlm.U.sum(0).mean())

    vlm.perform_PCA()
    vlm.knn_imputation(n_pca_dims=20, k=50, balanced=True, b_sight=300, b_maxl=1500, n_jobs=16)

    vlm.fit_gammas()
    #vlm.plot_phase_portraits([
    #    'DDIT3',
    #    'CDK1',
    #    ])

    vlm.predict_U()
    vlm.calculate_velocity()
    vlm.calculate_shift(assumption="constant_velocity")
    vlm.extrapolate_cell_at_t(delta_t=1.)

    # Get embedding
    vlm.ts = metadata_felix[['tsne1_5plus_dengue_reads', 'tsne2_5plus_dengue_reads']].fillna(0).values

    vlm.estimate_transition_prob(hidim="Sx_sz", embed="ts", transform="sqrt", psc=1,
                             n_neighbors=500, knn_random=True, sampled_fraction=0.5)
    vlm.calculate_embedding_shift(sigma_corr=0.05, expression_scaling=True)

    # Plot RNA velocity field
    vs = vlm.ts.copy()
    dvs = vlm.delta_embedding.copy()
    genes = ['virus_reads_per_million', 'DDIT3', 'clusterN_SNV', 'time [h]']
    fig, axs = plt.subplots(2, 2, figsize=(5, 4.5), sharex=True, sharey=True)
    axs = axs.ravel()
    for i in range(len(axs)):
        ax = axs[i]
        gene = genes[i]
        if gene in vlm.ra['Gene']:
            c = 1.0 * vlm.S[vlm.ra['Gene'] == gene][0]
            alpha = 0.6
        else:
            c = ds.samplesheet[gene].values
            alpha = 0.9
        if 'cluster' not in gene:
            c = np.log10(0.1 + c)
        c = (c - c.min()) / (c.max() - c.min())
        ax.scatter(vs[:, 0], vs[:, 1], c=c, alpha=0.3, s=10)
        ax.quiver(vs[:, 0], vs[:, 1], dvs[:, 0], dvs[:, 1], alpha=alpha)
        ax.set_title(gene)
    fig.tight_layout(h_pad=1, w_pad=0)


    # Alternative embedding
    from sklearn.manifold import TSNE
    bh_tsne = TSNE()
    vlm.ts = bh_tsne.fit_transform(vlm.pcs[:, :25])
    vlm.estimate_transition_prob(hidim="Sx_sz", embed="ts", transform="sqrt", psc=1,
                             n_neighbors=500, knn_random=True, sampled_fraction=0.5)
    vlm.calculate_embedding_shift(sigma_corr=0.05, expression_scaling=True)

    # Plot RNA velocity field
    vs = vlm.ts.copy()
    dvs = vlm.delta_embedding.copy()
    genes = ['virus_reads_per_million', 'DDIT3', 'CDK1', 'clusterN_SNV', 'time [h]', 'AURKA']
    fig2, axs = plt.subplots(2, 3, figsize=(7, 4.5), sharex=True, sharey=True)
    axs = axs.ravel()
    for i in range(len(axs)):
        ax = axs[i]
        gene = genes[i]
        if gene in vlm.ra['Gene']:
            c = 1.0 * vlm.S[vlm.ra['Gene'] == gene][0]
            alpha = 0.6
        else:
            c = ds.samplesheet[gene].values
            alpha = 0.9
        if 'cluster' not in gene:
            c = np.log10(0.1 + c)
        c = (c - c.min()) / (c.max() - c.min())
        ax.scatter(vs[:, 0], vs[:, 1], c=c, alpha=0.3, s=10)
        ax.quiver(vs[:, 0], vs[:, 1], dvs[:, 0], dvs[:, 1], alpha=alpha)
        ax.set_title(gene)
        ax.set_axis_off()
    fig2.text(0.52, 0.02, 'tSNE dimension 1', ha='center')
    fig2.text(0.02, 0.52, 'tSNE dimension 2', va='center', rotation=90)
    fig2.tight_layout(h_pad=1, w_pad=0)



    plt.ion()
    plt.show()

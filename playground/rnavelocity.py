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
import xarray as xr
import velocyto as vcy
import matplotlib.pyplot as plt
import seaborn as sns

os.environ['SINGLET_CONFIG_FILENAME'] = 'singlet.yml'
sys.path.append('/home/fabio/university/postdoc/singlet')
from singlet.dataset import Dataset


if __name__ == '__main__':

    print('Load RNA velocity results (loom file)')
    # NOTE: there is a mess trying to connect the velocyto output with the normal
    # htseq-count output, below is the explanation how I solved this.
    # velocyto skips quasi-empty BAM files, final order is lexicographic as in:
    # echo '' > ~/subfolders.tsv; for fdn in 10017006*; do si=$(du $fdn/star/Aligned.out.possorted.bam | cut -f1); if [ $si -ge "30" ]; then echo $fdn >> ~/subfolders.tsv; fi; done
    cellnames = pd.read_csv('../bigdata/rnavelocity_cellnames.tsv', header=None).values[:, 0]
    vlm = vcy.VelocytoLoom("../bigdata/rna_velocity.loom")
    vlm.ca['CellID'] = cellnames

    print('Load normal counts and metadata and sync them with the velocity results')
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

    print('Filter cells, genes, etc. using the velocity tutorial')
    vlm.normalize("S", size=True, log=True)
    vlm.filter_cells(bool_array=vlm.initial_Ucell_size > np.percentile(vlm.initial_Ucell_size, 0.5))
    ds.samplesheet = ds.samplesheet.loc[vlm.ca['CellID']]
    metadata_felix = metadata_felix.loc[vlm.ca['CellID']]

    vlm.score_detection_levels(min_expr_counts=40, min_cells_express=30)
    vlm.filter_genes(by_detection_levels=True)
    vlm.score_cv_vs_mean(3000, plot=True, max_expr_avg=35)
    vlm.filter_genes(by_cv_vs_mean=True)

    print('Velocity model normalization')
    vlm._normalize_S(relative_size=vlm.S.sum(0),
        target_size=vlm.S.sum(0).mean())
    vlm._normalize_U(relative_size=vlm.U.sum(0),
        target_size=vlm.U.sum(0).mean())

    print('Velocity PCA and knn')
    vlm.perform_PCA()
    vlm.knn_imputation(n_pca_dims=20, k=50, balanced=True, b_sight=300, b_maxl=1500, n_jobs=16)

    print('Velocity fit gammas')
    vlm.fit_gammas()
    #vlm.plot_phase_portraits([
    #    'DDIT3',
    #    'CDK1',
    #    ])

    print('Velocity predict unspliced, calculate velocity, shift')
    vlm.predict_U()
    vlm.calculate_velocity()
    vlm.calculate_shift(assumption="constant_velocity")
    vlm.extrapolate_cell_at_t(delta_t=1.)

    print('tSNE embedding')
    # Alternative embedding
    from sklearn.manifold import TSNE
    bh_tsne = TSNE()
    vlm.ts = bh_tsne.fit_transform(vlm.pcs[:, :25])

    print('Estimate shifts onto embedding')
    vlm.estimate_transition_prob(hidim="Sx_sz", embed="ts", transform="sqrt", psc=1,
                             n_neighbors=500, knn_random=True, sampled_fraction=0.5)
    vlm.calculate_embedding_shift(sigma_corr=0.05, expression_scaling=True)

    print('Plot tSNE of gene expression, viral load, and velocity')
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
    #fig2.savefig('../figures/tSNE_rnavelocity_overdispersed_infectioncellcycle.png')

    print('Export RNA velocity results')
    vse = pd.DataFrame(
            data=np.hstack([vs, dvs]),
            index=ds.samplenames,
            columns=['tsne 1', 'tsne 2', 'delta tsne 1', 'delta tsne 2'],
            )
    #vse.to_csv('../data/velocity_tsne_projection.tsv', sep='\t', index=True)


    print('Integrate SNP frequency data')
    data_par = xr.open_dataset('../bigdata/allele_frequencies_parental.nc')
    data_sc = xr.open_dataset('../bigdata/allele_frequencies.nc')

    af_par = data_par['aaf_avg']
    ma_par = data_par['alt']
    afs = data_sc['aaf']

    # Manually select a few positions from earlier plots, they are are subset
    # of poss_all. We could use all, but this is a good list for now
    # See also: check_genomic_location_SNVs.py
    poss = np.array([
        398, 401, 1530, 2280, 2424, 2425, 2426, 2429, 4828,
        5046, 5558, 5561, 8554, 8577,
        ])


    # NOTE: there's a few cells that have SNV data but did not survive here, exclude them:
    # 1001700610_B18, 1001700610_O16, 1001700612_A2, 1001700612_F20, 1001700612_K20,
    # 1001700612_L21, 1001700612_L22, 1001700612_L3, 1001700612_M12, 1001700612_N14,
    # 1001700612_N3
    # vice versa, there's no cell that is listed here that is missing SNV data, good

    # Reload counts for spliced, to keep all genes
    vlm_all = vcy.VelocytoLoom("../bigdata/rna_velocity.loom")
    vlm_all.ca['CellID'] = cellnames
    vlm_all.normalize("S", size=True, log=True)
    ind_cells = np.array([x in vlm.ca['CellID'] for x in vlm_all.ca['CellID']])

    print('Same but with single SNPs')
    # Plot RNA velocity field
    vs = vlm.ts.copy()
    dvs = vlm.delta_embedding.copy()
    genes = ['virus_reads_per_million', 'coverage', 'DDIT3', 'ACTB', 'LDHA', 'SEC61G', 'CDK1', 'AURKA', 'clusterN_SNV', 'time [h]']
    samplenames = ds.samplenames
    fig2, axs = plt.subplots(4, 6, figsize=(14, 10.5), sharex=True, sharey=True)
    color_by = {}
    axs = axs.ravel()
    for i in range(len(axs)):
        ax = axs[i]
        if i < len(genes):
            gene = genes[i]
            if gene in vlm_all.ra['Gene']:
                c = 1.0 * vlm_all.S[vlm_all.ra['Gene'] == gene][0][ind_cells]
                alpha = 0.6
            # Categorical
            else:
                c = ds.samplesheet[gene].values
                alpha = 0.7
            if gene not in ('clusterN_SNV', 'time [h]'):
                c = np.log10(0.1 + c)
                if gene not in ('coverage',):
                    c = (c - c.min()) / (c.max() - c.min())
            else:
                c_unique = np.unique(c).tolist()
                palette = sns.color_palette('Set2', n_colors=len(c_unique))
                c = [palette[c_unique.index(x)] for x in c]
                color_by[gene] = dict(zip(c_unique, palette))
        elif i - len(genes) < len(poss):
            pos = poss[i-len(genes)]
            gene = 'SNV {:}'.format(pos)
            c = afs.sel(sample=samplenames, position=pos).fillna(-1).data
            ind_miss = c < 0

            # Take logit of nonneg values to expand the dynamic range
            # Then renormalize between 0 and 1
            def logit(x, ext=1e-3):
                return -np.log10(1. / np.minimum(np.maximum(x, ext), 1 - ext) - 1)
            c[~ind_miss] = (logit(c[~ind_miss]) + 3) / 6

            # Color by allele frequency, put missing data as light grey
            cmapname = 'plasma'
            cmap = plt.cm.get_cmap(cmapname)
            c = cmap(c)
            c[ind_miss] = [0.8, 0.8, 0.8, 1]
            alpha = 0.6
        else:
            continue

        ax.scatter(vs[:, 0], vs[:, 1], c=c, alpha=0.3, s=10)
        ax.quiver(vs[:, 0], vs[:, 1], dvs[:, 0], dvs[:, 1], alpha=alpha, width=0.005)
        ax.set_title(gene)
        ax.set_axis_off()
    fig2.text(0.52, 0.14, 'tSNE dimension 1', ha='center')
    fig2.text(0.02, 0.52, 'tSNE dimension 2', va='center', rotation=90)
    fig2.tight_layout(h_pad=1, w_pad=0, rect=(0, 0.15, 1, 1))
    # Add color maps
    import matplotlib.patches as mpatches
    ax = fig2.add_axes((0, 0.07, 1, 0.04))
    ax.set_axis_off()
    handles = [mpatches.Patch(color=val, label=str(key)) for key, val in color_by['clusterN_SNV'].items()]
    ax.legend(handles=handles, ncol=len(handles), loc='upper center', title='Cluster')
    ax = fig2.add_axes((0, 0.02, 1, 0.04))
    ax.set_axis_off()
    handles = [mpatches.Patch(color=val, label=str(key)) for key, val in color_by['time [h]'].items()]
    ax.legend(handles=handles, ncol=len(handles), loc='upper center', title='Time [h]')
    #fig2.savefig('../figures/tSNE_rnavelocity_overdispersed_infectioncellcycle_withSNVs.png')


    if False:
        print('Get tSNE embedding from outside (tSNE on DENV SNVs)')
        vlm.ts = metadata_felix[['tsne1_5plus_dengue_reads', 'tsne2_5plus_dengue_reads']].fillna(0).values

        print('Estimate shifts onto external embedding')
        vlm.estimate_transition_prob(hidim="Sx_sz", embed="ts", transform="sqrt", psc=1,
                                 n_neighbors=500, knn_random=True, sampled_fraction=0.5)
        vlm.calculate_embedding_shift(sigma_corr=0.05, expression_scaling=True)

        print('Plot results')
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

    # Find the two clouds
    dso = Dataset(
            counts_table='dengue',
            samplesheet='dengue',
            featuresheet='humanGC38',
            )
    dso.query_samples_by_name(ds.samplenames, inplace=True)
    dso.counts.normalize(inplace=True)
    dso.feature_selection.unique(inplace=True)
    dso.reindex(axis='features', column='GeneName', inplace=True, drop=False)
    dso.samplesheet['virus_reads_per_million'] = ds.samplesheet['virus_reads_per_million']

    ind_2424 = afs.sel(sample=samplenames, position=2424).fillna(-1).data >= 0.1
    ind_not2424 = afs.sel(sample=samplenames, position=2424).fillna(-1).data < 0.1
    dso.samplesheet['is_2424'] = ind_2424
    dsp = dso.split('is_2424')
    fig, ax = plt.subplots(figsize=(3.8, 3.2))
    colors = {True: 'steelblue', False: 'darkred'}
    for key, dsi in dsp.items():
        color = colors[key]
        if key is True:
            label = 'M2'
        else:
            label = 'M1'
        y = np.log10(0.1 + dsi.counts.loc['DDIT3'].values)
        x = np.log10(0.1 + np.random.normal(0, 0.1, size=len(y)) + dsi.samplesheet['virus_reads_per_million'].values)
        ax.scatter(x, y, s=10, color=color, alpha=0.15, label=label)
    ax.grid(True)
    ax.legend(loc='upper left', title='Mutant:')
    ax.set_xticks([-1, 1, 3, 5])
    ax.set_yticks([-1, 1, 3, 5])
    ax.set_xticklabels(['$0$', '$10$', '$10^3$', '$10^5$'])
    ax.set_yticklabels(['$0$', '$10$', '$10^3$', '$10^5$'])
    ax.set_xlim(0.9, 6)
    ax.set_ylim(-1.1, 5)
    ax.set_ylabel('DDIT3 expression [cpm]')
    ax.set_xlabel('vRNA expression [cpm]')
    fig.tight_layout()
    fig.savefig('/home/fabio/university/PI/grants/AU_Ideas_2019/figures/DDIT3_corr_mutants.svg')


    plt.ion()
    plt.show()

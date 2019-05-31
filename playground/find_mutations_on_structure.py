# vim: fdm=indent
'''
author:     Fabio Zanini
date:       18/03/19
content:    Find common (parental) mutations on 3D structure.
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



# Script
if __name__ == '__main__':

    #print('Load SNP data')
    #data_par = xr.open_dataset('../bigdata/allele_frequencies_parental.nc')
    #data_sc = xr.open_dataset('../bigdata/allele_frequencies.nc')

    #af_par = data_par['aaf_avg']
    #ma_par = data_par['alt']
    #afs = data_sc['aaf']


    ## Manually select a few positions from earlier plots, they are are subset
    ## of poss_all. We could use all, but this is a good list for now
    #poss = np.array([
    #    398, 401, 1530, 2280, 2424, 2425, 2426, 2429, 4828,
    #    5046, 5558, 5561, 8554, 8577,
    #    ])

    print('Load annotated mutations (syn/nonsyn, context, etc.)')
    mutations = pd.read_csv(
            '../data/mutations_highvariance_summary.tsv',
            sep='\t',
            index_col=0)

    print('Load annotated reference')
    from Bio import SeqIO
    from Bio.Seq import translate
    ref_fn = '../data/U87411_strain16681.gb'
    ref = SeqIO.read(ref_fn, format='gb')
    proteins = []
    names = ['C', 'pr', 'M', 'E', 'NS1', 'NS2A', 'NS2B', 'NS3', 'NS4A', '2K', 'NS4B', 'NS5']
    for fea in ref.features:
        # Keep only proteins
        if fea.type == 'mat_peptide':
            name = fea.qualifiers['product'][0]
            # Some mutations are in the anchor region of C
            if name == 'capsid protein C':
                continue
            for n in names:
                if name.endswith(n):
                    print('Found {:}: {:}'.format(n, name))
                    break
            else:
                continue
            prot = {
                'name': n,
                'start': fea.location.nofuzzy_start,
                'end': fea.location.nofuzzy_end,
                'seq': str(fea.extract(ref).seq),
                'seq_aa': translate(str(fea.extract(ref).seq)),
                }
            proteins.append(prot)

    print('Assign proteins to mutations')
    mutations['protein'] = ''
    mutations['pos_in_prot'] = -1
    for m, row in mutations.iterrows():
        pos = row['position']
        for prot in proteins:
            if (pos >= prot['start']) and (pos < prot['end']):
                mutations.at[m, 'protein'] = prot['name']
                mutations.at[m, 'pos_in_prot'] = (pos - prot['start']) // 3
                break

    print('Load structures and align mutations on them')
    from Bio.PDB import PDBParser
    from seqanpy import align_overlap, align_local
    pa = PDBParser()
    fdn = '../data/structures/'
    fns = {
        #'NS1': 'NS1_4oig.pdb',
        'NS1': 'NS1_4o6b.pdb',
        'NS3': 'NS3_5yw1.pdb',
        'NS5': 'NS5_5i3q.pdb',
        'C': 'virion_3j27.pdb',
        'M': 'virion_3j27.pdb',
        'E': 'virion_3j27.pdb',
        }
    d3to1 = {
            'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
            'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N',
            'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W',
            'ALA': 'A', 'VAL': 'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M',
            # This is water??
            'HOH': 'O',
            }
    mutations['PDB_fn'] = ''
    mutations['PDB_id'] = ''
    mutations['PDB_chain'] = ''
    mutations['PDB_pos_in_chain'] = -1
    mutations['PDB_allele'] = ''

    for protname in ['NS3', 'NS5', 'E', 'NS1']:
        print(protname)
        for prot in proteins:
            if prot['name'] == protname:
                break
        muts = mutations.loc[mutations['protein'] == protname]

        # Get the structure
        struc = pa.get_structure(protname, fdn+fns[protname])
        # Get the right chain
        scores = []
        for chain in struc.get_chains():
            # I is the compound (inhibitors) used for crystallization
            if chain.id == 'I':
                continue
            #print('{:}, chain {:}'.format(protname, chain.id))
            seql = [d3to1.get(r.get_resname(), 'O') for r in chain.get_residues()]
            seq = ''.join(seql)
            s, a1, a2 = align_local(prot['seq_aa'], seq)
            scores.append(s)
        chain = list(struc.get_chains())[np.argmax(s)]
        seql = [d3to1.get(r.get_resname(), 'O') for r in chain.get_residues()]
        seq = ''.join(seql)

        # Flag all mutations
        for m, mut in muts.iterrows():
            mutations.at[m, 'PDB_fn'] = fns[protname]
            mutations.at[m, 'PDB_id'] = fns[protname].split('_')[1].upper().split('.')[0]
            mutations.at[m, 'PDB_chain'] = chain.id

            s, a1, a2 = align_overlap(seq, mut['context_protein'])
            # The focal allele is always small and is the only such letter
            pos_in_context = 4
            pos = a2.find(mut['context_protein'][pos_in_context])
            pos -= a1[:pos].count('-')
            mutations.at[m, 'PDB_pos_in_chain'] = pos
            mutations.at[m, 'PDB_allele'] = seq[pos]
            print('Ref: '+mut['context_protein'])
            print('PDB: '+seq[pos-4:pos]+seq[pos].lower()+seq[pos+1: pos+5])

    #mutations.to_csv('../data/mutations_highvariance_summary.tsv', sep='\t', index=True)

    print('Load multiple sequence alignments')
    from Bio import AlignIO
    protname = 'NS1'
    ali = AlignIO.read('../data/msa/denv2_{:}.fasta'.format(protname), 'fasta')
    # NS1 has no gaps, go ahead
    aalphal = list(d3to1.values())
    aalpha = np.array(aalphal)

    # Translate alignment
    alip = []
    for seq in ali:
        alip.append(list(seq.seq.translate()))
    alim = np.array(alip)

    aafs = np.zeros((len(aalpha), alim.shape[1]), float)
    for ia, aa in enumerate(aalpha):
        aafs[ia] = (alim == aa).mean(axis=0)
    ent = -(np.log2(aafs + 1e-10) * aafs).sum(axis=0)
    cons = aalpha[aafs.argmax(axis=0)]
